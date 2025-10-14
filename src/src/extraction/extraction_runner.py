import torch
import random
import math
import copy
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import models.extraction
from tqdm import tqdm
from pathlib import Path


class Classification(torch.nn.Module):

    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()

        self.fc1 = torch.nn.Linear(emb_size, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


def extract_outputs(graph_data, specific_nodes, independent_model, surrogate_model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    independent_model.eval()
    surrogate_model.eval()
    
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    independent_embedding, independent_logits = independent_model(input_data)
    surrogate_embedding, surrogate_logits = surrogate_model(input_data)

    softmax = torch.nn.Softmax(dim=1)
    independent_prob = softmax(independent_logits)
    surrogate_prob = softmax(surrogate_logits)

    if specific_nodes != None:
        independent_prob = independent_prob[specific_nodes].cpu()
        surrogate_prob = surrogate_prob[specific_nodes].cpu()
        independent_embedding = independent_embedding[specific_nodes].cpu()
        surrogate_embedding = surrogate_embedding[specific_nodes].cpu()
        independent_logits = independent_logits[specific_nodes].cpu()
        surrogate_logits = surrogate_logits[specific_nodes].cpu()

    probability = {'independent': independent_prob, 'surrogate': surrogate_prob}
    embedding = {'independent': independent_embedding, 'surrogate': surrogate_embedding}
    logits = {'independent': independent_logits, 'surrogate': surrogate_logits}

    return probability, logits, embedding


def verify(suspicious_logits, verifier_model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    distance = torch.flatten(suspicious_logits).view(1, -1)

    verifier_model.to(device)
    verifier_model.eval()

    outputs = verifier_model(distance.to(device))

    return outputs


def evaluate_target_response(args, graph_data, model, response, process):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.eval()
    model = model.to(device)

    if args.task_type == 'transductive':    
        input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
        embedding, output = model(input_data)
        embedding = embedding.detach()
        output = output.detach()

        if process == 'train':
            search_nodes_index = graph_data.shadow_nodes_index
        elif process == 'test':
            search_nodes_index = graph_data.attacker_nodes_index

        if response == 'train_embeddings':
            target_response = embedding[search_nodes_index]
        elif response == 'train_outputs':
            target_response = output[search_nodes_index]
        elif response == 'test_embeddings':
            target_response = embedding[graph_data.test_nodes_index]
        elif response == 'test_outputs':
            target_response = output[graph_data.test_nodes_index]
    elif args.task_type == 'inductive':
        if process == 'train':
            extraction_input_data = graph_data[1].features.to(device), graph_data[1].adjacency.to(device)
            extraction_embedding, extraction_output = model(extraction_input_data)
            extraction_embedding = extraction_embedding.detach()
            extraction_output = extraction_output.detach()
        elif process == 'test':
            extraction_input_data = graph_data[2].features.to(device), graph_data[2].adjacency.to(device)
            extraction_embedding, extraction_output = model(extraction_input_data)
            extraction_embedding = extraction_embedding.detach()
            extraction_output = extraction_output.detach()

        test_input_data = graph_data[3].features.to(device), graph_data[3].adjacency.to(device)
        test_embedding, test_output = model(test_input_data)
        test_embedding = test_embedding.detach()
        test_output = test_output.detach()

        if response == 'train_embeddings':
            target_response = extraction_embedding
        elif response == 'train_outputs':
            target_response = extraction_output
        elif response == 'test_embeddings':
            target_response = test_embedding
        elif response == 'test_outputs':
            target_response = test_output
        
        
    return target_response


def train_extraction_model(args, model_save_path, data, process, classifier):
    clf_save_path = model_save_path + '_clf.pt'
    graph_data, train_emb, train_outputs, test_outputs = data
    softmax = torch.nn.Softmax(dim=1)
    train_outputs = softmax(train_outputs)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # prepare model
    if args.task_type == 'transductive':
        in_dim = graph_data.feat_dim
    elif args.task_type == 'inductive':
        in_dim = graph_data[1].feat_dim

    if args.extraction_method == 'white_box':
        out_dim = train_emb.shape[1]
    elif args.extraction_method == 'black_box':
        out_dim = train_outputs.shape[1]

    if args.extraction_model == 'gcn':
        extraction_model = models.extraction.GcnExtract(in_dim, out_dim, hidden_dim=args.extraction_hidden_dim)
    elif args.extraction_model == 'sage':
        extraction_model = models.extraction.SageExtract(in_dim, out_dim, hidden_dim=args.extraction_hidden_dim)
    elif args.extraction_model == 'gat':
        extraction_model = models.extraction.GatExtract(in_dim, out_dim, hidden_dim=args.extraction_hidden_dim)
    elif args.extraction_model == 'gin':
        extraction_model = models.extraction.GinExtract(in_dim, out_dim, hidden_dim=args.extraction_hidden_dim)
    elif args.extraction_model == 'sgc':
        extraction_model = models.extraction.SGCExtract(in_dim, out_dim, hidden_dim=args.extraction_hidden_dim)

    extraction_model = extraction_model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer_medium = torch.optim.Adam(extraction_model.parameters(), lr=args.extraction_lr)

    clf = None
    if args.extraction_method == 'white_box':
        if args.task_type == "inductive":
            clf = Classification(out_dim, graph_data[0].class_num)
        else:
            clf = Classification(out_dim, graph_data.class_num)
        clf = clf.to(device)
        optimizer_classification = torch.optim.SGD(clf.parameters(), lr=args.extraction_lr)
    elif args.extraction_method == 'black_box':
        clf = None
    predict_fn = lambda output: output.max(1, keepdim=True)[1]

    # train extraction model
    last_train_acc, last_train_fide = 0.0, 0.0
    if args.task_type == 'transductive':
        path = Path(model_save_path)
        if path.is_file():
            extraction_model = torch.load(model_save_path)
            if args.extraction_method == 'white_box':
                clf = torch.load(clf_save_path)
        else:
            if process == 'train':
                search_nodes_index = graph_data.shadow_nodes_index
            elif process == 'test':
                search_nodes_index = graph_data.attacker_nodes_index
            
            for epoch in range(args.extraction_train_epochs):
                extraction_model.train()
                if args.extraction_method == 'white_box':
                    clf.train()
                train_emb = train_emb.to(device)
                train_outputs = train_outputs.to(device)
                input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
                extraction_embeddings, extraction_outputs = extraction_model(input_data)
                part_embeddings = extraction_embeddings[search_nodes_index]
                part_outputs = extraction_outputs[search_nodes_index]

                if args.extraction_method == 'white_box':
                    optimizer_medium.zero_grad()
                    optimizer_classification.zero_grad()
                    loss_emb = torch.sqrt(loss_fn(part_embeddings, train_emb))
                    loss_emb.backward()
                    optimizer_medium.step()

                    outputs = clf(part_embeddings.detach())
                    train_labels = predict_fn(train_outputs)
                    train_labels = torch.flatten(train_labels)
                    loss_out = loss_fn(outputs, train_labels)
                    loss_out.backward()
                    optimizer_classification.step()
                elif args.extraction_method == 'black_box':
                    optimizer_medium.zero_grad()
                    loss = loss_fn(part_outputs, train_outputs)
                    if process == 'test' and classifier != None:
                        surrogate_outputs, _, _ = extract_outputs(graph_data, graph_data.target_nodes_index, extraction_model, extraction_model)
                        classify_logits = verify(surrogate_outputs["surrogate"], classifier)
                        classify_logits = torch.flatten(classify_logits)
                        evade_loss = loss_fn(classify_logits, torch.tensor(0).to(device))
                        loss += 10 * evade_loss

                    loss.backward()
                    optimizer_medium.step()

                if (epoch + 1) % 50 == 0:
                    extraction_model.eval()
                    if args.extraction_method == 'white_box':
                        clf.eval()

                    acc_correct = 0
                    fide_correct = 0

                    embeddings, outputs = extraction_model(input_data)
                    if args.extraction_method == 'white_box':
                        outputs = clf(embeddings.detach())
                    pred = predict_fn(outputs)
                    train_labels = predict_fn(train_outputs)
                
                    for i in range(len(search_nodes_index)):
                        if pred[search_nodes_index[i]] == graph_data.labels[search_nodes_index[i]]:
                            acc_correct += 1
                        if pred[search_nodes_index[i]] == train_labels[i]:
                            fide_correct += 1

                    accuracy = acc_correct * 100.0 / len(search_nodes_index)
                    fidelity = fide_correct * 100.0 / train_outputs.shape[0]
                    if last_train_acc == 0.0 or last_train_fide == 0.0:
                        last_train_acc = accuracy
                        last_train_fide = fidelity
                    else:
                        train_acc_diff = (accuracy - last_train_acc) / last_train_acc * 100
                        train_fide_diff = (fidelity - last_train_fide) / last_train_fide * 100
                        if train_acc_diff <= 0.5 and train_fide_diff <= 0.5: # 0.5%
                            break
                        else:
                            last_train_acc = accuracy
                            last_train_fide = fidelity
            
            torch.save(extraction_model, model_save_path)
            if args.extraction_method == 'white_box':
                torch.save(clf, clf_save_path)

        extraction_model.eval()
        if args.extraction_method == 'white_box':
            clf.eval()
        acc_correct, fide_correct = 0, 0
        input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
        embeddings, outputs = extraction_model(input_data)
        if args.extraction_method == 'white_box':
            outputs = clf(embeddings.detach())
        pred = predict_fn(outputs)
        test_labels = predict_fn(test_outputs)
        for i in range(len(graph_data.test_nodes_index)):
            if pred[graph_data.test_nodes_index[i]] == graph_data.labels[graph_data.test_nodes_index[i]]:
                acc_correct += 1
            if pred[graph_data.test_nodes_index[i]] == test_labels[i]:
                fide_correct += 1
        accuracy = acc_correct * 100.0 / len(graph_data.test_nodes_index)
        fidelity = fide_correct * 100.0 / test_outputs.shape[0]
        save_acc = round(accuracy, 3)
        save_fide = round(fidelity, 3)
    elif args.task_type == 'inductive':
        path = Path(model_save_path)
        if path.is_file():
            extraction_model = torch.load(model_save_path)
            if args.extraction_method == 'white_box':
                clf = torch.load(clf_save_path)
        else:
            if process == 'train':
                using_graph_data = graph_data[1]
            elif process == 'test':
                using_graph_data = graph_data[2]
            
            for epoch in range(args.extraction_train_epochs):
                extraction_model.train()
                if args.extraction_method == 'white_box':
                    clf.train()
                train_emb = train_emb.to(device)
                train_outputs = train_outputs.to(device)

                input_data = using_graph_data.features.to(device), using_graph_data.adjacency.to(device)
                extraction_embeddings, extraction_outputs = extraction_model(input_data)

                if args.extraction_method == 'white_box':
                    optimizer_medium.zero_grad()
                    optimizer_classification.zero_grad()
                    loss_emb = torch.sqrt(loss_fn(extraction_embeddings, train_emb))
                    loss_emb.backward()
                    optimizer_medium.step()

                    outputs = clf(extraction_embeddings.detach())
                    train_labels = predict_fn(train_outputs)
                    train_labels = torch.flatten(train_labels)
                    loss_out = loss_fn(outputs, train_labels)
                    loss_out.backward()
                    optimizer_classification.step()
                elif args.extraction_method == 'black_box':
                    optimizer_medium.zero_grad()
                    loss = loss_fn(extraction_outputs, train_outputs)
                    loss.backward()
                    optimizer_medium.step()

                if (epoch + 1) % 50 == 0:
                    extraction_model.eval()
                    if args.extraction_method == 'white_box':
                        clf.eval()

                    acc_correct = 0
                    fide_correct = 0

                    embeddings, outputs = extraction_model(input_data)
                    if args.extraction_method == 'white_box':
                        outputs = clf(embeddings.detach())
                    pred = predict_fn(outputs)
                    train_labels = predict_fn(train_outputs)
                
                    for i in range(using_graph_data.node_num):
                        if pred[i] == using_graph_data.labels[i]:
                            acc_correct += 1
                        if pred[i] == train_labels[i]:
                            fide_correct += 1

                    accuracy = acc_correct * 100.0 / using_graph_data.node_num
                    fidelity = fide_correct * 100.0 / train_outputs.shape[0]
                    if last_train_acc == 0.0 or last_train_fide == 0.0:
                        last_train_acc = accuracy
                        last_train_fide = fidelity
                    else:
                        train_acc_diff = (accuracy - last_train_acc) / last_train_acc * 100
                        train_fide_diff = (fidelity - last_train_fide) / last_train_fide * 100
                        if train_acc_diff <= 0.5 and train_fide_diff <= 0.5: # 0.5%
                            break
                        else:
                            last_train_acc = accuracy
                            last_train_fide = fidelity
            
            torch.save(extraction_model, model_save_path)
            if args.extraction_method == 'white_box':
                torch.save(clf, clf_save_path)

        extraction_model.eval()
        if args.extraction_method == 'white_box':
            clf.eval()
        acc_correct, fide_correct = 0, 0
        input_data = graph_data[3].features.to(device), graph_data[3].adjacency.to(device)
        embeddings, outputs = extraction_model(input_data)
        if args.extraction_method == 'white_box':
            outputs = clf(embeddings.detach())
        pred = predict_fn(outputs)
        test_labels = predict_fn(test_outputs)
        for i in range(graph_data[3].node_num):
            if pred[i] == graph_data[3].labels[i]:
                acc_correct += 1
            if pred[i] == test_labels[i]:
                fide_correct += 1
        accuracy = acc_correct * 100.0 / graph_data[3].node_num
        fidelity = fide_correct * 100.0 / test_outputs.shape[0]
        save_acc = round(accuracy, 3)
        save_fide = round(fidelity, 3)

    return extraction_model, clf, save_acc, save_fide


def run(args, model_save_path, graph_data, original_model, process, classifier):
    train_emb = evaluate_target_response(args, graph_data, original_model, 'train_embeddings', process) # we do not use this in black-box extraction setting
    train_outputs = evaluate_target_response(args, graph_data, original_model, 'train_outputs', process)
    test_outputs = evaluate_target_response(args, graph_data, original_model, 'test_outputs', process)
    extraction_data = graph_data, train_emb, train_outputs, test_outputs
    extraction_model, _, extraction_acc, extraction_fide = train_extraction_model(args, model_save_path, extraction_data, process, classifier)

    return extraction_model, extraction_acc, extraction_fide



if __name__ == '__main__':
    pass
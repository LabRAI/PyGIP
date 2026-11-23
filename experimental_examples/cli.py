#Cli - cli commands
if __name__ == '__main__':
    import argparse,torch
    from dataset import Dataset
    from attacker import GNNFingersAttack, FingerprintSpecLocal
    import os
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='Cora')
    parser.add_argument('--joint_steps',type=int, default=300)
    parser.add_argument('--num_graphs',type=int, default=64)
    parser.add_argument('--num_nodes',type=int, default=32)
    parser.add_argument('--edge_density',type=float,default=0.05)
    parser.add_argument('--proj_every',type=int,default=25)
    parser.add_argument('--node_sample',type=int,default=0)
    parser.add_argument('--device',default=None)
    parser.add_argument('--mode',choices=['attack','defense'],default='attack',help='Run attack pipeline or defense pipeline')
    parser.add_argument('--clean', action='store_true', help='Remove old .pt and metrics files before running')
    args = parser.parse_args()

    if args.clean:
        print('[clean] removing old .pt and metrics files...')
        for f in glob.glob('*.pt') + glob.glob('*.json'):
            try:
                os.remove(f)
                print(' removed',f)
            except Exception as e:
                print(' could not remove',f,e)

    ds = Dataset(api_type='pyg', path='./data', name=args.dataset)
    ds.to(torch.device(args.device) if args.device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))

    fp_cfg = FingerprintSpecLocal(num_graphs = args.num_graphs,num_nodes = args.num_nodes,edge_density = args.edge_density,proj_every = args.proj_every,node_sample = args.node_sample)
    attack = GNNFingersAttack(ds, attack_node_fraction = 0.3, fp_cfg = fp_cfg,joint_steps = args.joint_steps, device = args.device)

    if args.mode == 'attack':
        metrics = attack.attack()
        print('\nSummary:')
        print('ROC_AUC:', metrics['ROC_AUC'])
        print('ARUC:', metrics['ARUC'])
        print('Saved artifacts: univerifier.pt, fingerprints.pt, verification_metrics.json')
    else:   # defense
        res = attack.defense(method='default')
        print('\nDefense result:')
        print('quick metrics:', res.get('metrics'))
        print('Saved defended model:', './defended_model.pt' if res.get('defense_model') is not None else 'none')

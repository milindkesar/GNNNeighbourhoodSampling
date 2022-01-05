from graphsage.model import run_general
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",type=str)
parser.add_argument("--epochs",type=int, default=200)
parser.add_argument("--rand_walk")
parser.add_argument("--n_layers",type=int)
parser.add_argument("--neighbours1", type=int, default=20)
parser.add_argument("--neighbours2",type=int,default=20)
parser.add_argument("--aggregator",default="mean", choices = ["mean","weighted_mean"])
parser.add_argument("--attention",default="normal",choices=["normal","softmax"])
parser.add_argument("--iter",type=int,default=1)
parser.add_argument("--exp",default='default')
parser.add_argument("--lr",default=0.01,type=float)
parser.add_argument("--outdir",default=None)
parser.add_argument("--includenodefeats",default="no")
parser.add_argument('--typewalk',default='default')
parser.add_argument('--p',default=1,type=float)
parser.add_argument('--q',default=1,type=float)
parser.add_argument('--num_walks',default=10,type=int)
parser.add_argument('--walk_length',default=10,type=int)
parser.add_argument('--teleport',default=0.2,type=float)
parser.add_argument('--teleport_khop')
parser.add_argument('--dfactor',default=2,type=float)
args = parser.parse_args()
def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass  #error condition maybe?
if __name__ == "__main__":
    rw=t_or_f(args.rand_walk)
    teleport_khop = t_or_f(args.teleport_khop)
    #print(args.dataset,args.epochs,rw,args.neighbours1,args.neighbours2,args.aggregator,args.iter,args.outdir)
    #print(args.typewalk)
    print(args.p,args.q)
    run_general(name=args.dataset,outdir=args.outdir,rw=rw,neighbours1=args.neighbours1,neighbours2=args.neighbours2,aggregator=args.aggregator,epochs=args.epochs,random_iter=args.iter,n_layers=args.n_layers, attention=args.attention,lr=args.lr,includenodefeats=args.includenodefeats,type_of_walk=args.typewalk, p=args.p, q=args.q, num_walks=args.num_walks, walk_length=args.walk_length, teleport=args.teleport, teleport_khop=teleport_khop, dfactor=args.dfactor)
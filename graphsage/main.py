from graphsage.model import run_wiki_cs,run_general
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
    #print(args.dataset,args.epochs,rw,args.neighbours1,args.neighbours2,args.aggregator,args.iter,args.outdir)
    run_general(name=args.dataset,outdir=args.outdir,rw=rw,neighbours1=args.neighbours1,neighbours2=args.neighbours2,aggregator=args.aggregator,epochs=args.epochs,random_iter=args.iter,n_layers=args.n_layers, attention=args.attention,lr=args.lr,includenodefeats=args.includenodefeats)
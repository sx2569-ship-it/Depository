import argparse
import glob
import os
from stitching import StitchConfig, stitch_images

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",type=str,required=True,help="Folder containing input images")
    ap.add_argument("--out_dir",type=str,required=True,help="Output folder")
    ap.add_argument("--feature",type=str,default="ORB",choices=["ORB", "SIFT"])
    ap.add_argument("--blend",type=str,default="feather",choices=["feather", "seam_multiband"])
    ap.add_argument("--scale",type=float,default=0.15)
    ap.add_argument("--max_features",type=int,default=1200)
    ap.add_argument("--levels",type=int,default=4)
    ap.add_argument("--tag",type=str,default="demo")
    args=ap.parse_args()
    paths=sorted(glob.glob(os.path.join(args.data_dir, "*.JPG")))
    if not paths:
        raise FileNotFoundError(f"No .JPG images found in {args.data_dir}")
    cfg=StitchConfig(
        feature=args.feature,
        blend=args.blend,
        resize_scale=args.scale,
        max_features=args.max_features,
        multiband_levels=args.levels
    )
    meta=stitch_images(paths,cfg,out_dir=args.out_dir,tag=args.tag)
    print("Saved panorama to:",os.path.join(args.out_dir,f"{args.tag}_pano.jpg"))
    print("Saved run log to:",os.path.join(args.out_dir,f"{args.tag}_meta.json"))
    print("Metrics:",meta["metrics"])

if __name__ == "__main__":
    main()

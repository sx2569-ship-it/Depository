import argparse
import os
import glob
import json
import pandas as pd
from stitching import StitchConfig, stitch_images

def run_one(paths,out_dir,tag,feature,blend,scale,max_features,levels):
    cfg=StitchConfig(
        feature=feature,
        blend=blend,
        resize_scale=scale,
        max_features=max_features,
        multiband_levels=levels,
    )
    meta=stitch_images(paths,cfg,out_dir=out_dir,tag=tag)
    return meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",required=True, help="Folder containing input .JPG images")
    ap.add_argument("--out_root",default="out",help="Output root folder")
    ap.add_argument("--scale",type=float,default=0.15)
    ap.add_argument("--orb_features",type=int,default=1000)
    ap.add_argument("--sift_features",type=int,default=1200)
    ap.add_argument("--levels",type=int,default=4)
    args=ap.parse_args()
    paths=sorted(glob.glob(os.path.join(args.data_dir, "*.JPG")))
    if not paths:
        raise FileNotFoundError(f"No .JPG images found in {args.data_dir}")
    results_dir=os.path.join(args.out_root,"results")
    figures_dir=os.path.join(args.out_root,"figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    metas=[]
    metas.append(run_one(paths,results_dir,"orb_feather","ORB","feather",args.scale,args.orb_features,args.levels))
    metas.append(run_one(paths,results_dir,"orb_seammb","ORB","seam_multiband",args.scale, args.orb_features,args.levels))
    metas.append(run_one(paths,results_dir,"sift_feather","SIFT","feather",args.scale,args.sift_features,args.levels))
    metas.append(run_one(paths,results_dir,"sift_seammb", "SIFT", "seam_multiband",args.scale,args.sift_features,args.levels))

    # Summarize to CSV
    rows=[]
    for m in metas:
        metr=m["metrics"]
        rows.append({
            "Setting":m["tag"].replace("_"," + ").upper(),
            "Mean SSIM":round(metr["mean_ssim"],3),
            "Mean PSNR (dB)":round(metr["mean_psnr"],2),
            "Seam Energy":round(metr["mean_seam_energy"],2),
            "Runtime (s)":round(metr["runtime_sec"],2),
            "Mean Inliers":round(metr["mean_inliers"],1),
            "Canvas (H×W)":f"{m['out_shape'][0]}×{m['out_shape'][1]}",
        })
    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir,"metrics_summary.csv"),index=False)
        # Add valid ratio (fraction of non-zero pixels in final panorama mask)
    from scripts.compute_valid_ratio import compute_valid_ratio_for_results
    df2=compute_valid_ratio_for_results(results_dir,df)
    df2.to_csv(os.path.join(results_dir,"metrics_summary_with_valid_ratio.csv"),index=False)
        # Generate figures for report/ppt
    from scripts.make_report_figures import generate_all_figures
    generate_all_figures(data_dir=args.data_dir,results_dir=results_dir, out_dir=figures_dir)
    print("Done.")
    print("Results:", results_dir)
    print("Figures:", figures_dir)

if __name__ == "__main__":
    main()

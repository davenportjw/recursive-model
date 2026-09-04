#!/usr/bin/env python3
"""
CLI utility to dispatch Tiny Recursive Gemma training jobs to Google Cloud Vertex AI.
Target Project: davenport-boutique
Target Region:  us-central1
Target Compute: Dedicated Serverless GPU (NVIDIA L4 24GB or NVIDIA T4 16GB)
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Dict, Any, Optional

DEFAULT_PROJECT = "davenport-boutique"
DEFAULT_REGION = "us-central1"
DEFAULT_BUCKET = "davenport-boutique-vertex-staging"
DEFAULT_CONTAINER = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"

GPU_CONFIGS = {
    "L4": {
        "accelerator_type": "NVIDIA_L4",
        "machine_type": "g2-standard-4",
        "accelerator_count": 1,
        "vram": "24GB"
    },
    "T4": {
        "accelerator_type": "NVIDIA_TESLA_T4",
        "machine_type": "n1-standard-4",
        "accelerator_count": 1,
        "vram": "16GB"
    },
    "CPU": {
        "accelerator_type": "ACCELERATOR_TYPE_UNSPECIFIED",
        "machine_type": "c2-standard-4",
        "accelerator_count": 0,
        "vram": "None"
    }
}

def generate_job_spec(
    display_name: str,
    gcs_source_uri: str,
    gpu_type: str = "L4",
    model_id: str = "google/gemma-4-E2B-it-qat-q4_0-unquantized",
    epochs: int = 3,
    batch_size: int = 4,
    iterations: int = 3,
    reasoning_steps: int = 2,
    decay_gamma: float = 1.5,
    act: bool = True,
    output_bucket: str = DEFAULT_BUCKET
) -> Dict[str, Any]:
    gpu_info = GPU_CONFIGS.get(gpu_type.upper(), GPU_CONFIGS["L4"])
    
    # Startup and training command inside the pre-built PyTorch container
    train_command = [
        "bash", "-c",
        f"set -e\n"
        f"echo '[Vertex Cloud Runner] Unpacking training source package...'\n"
        f"mkdir -p /app && cd /app\n"
        f"gsutil cp {gcs_source_uri} source.tar.gz\n"
        f"tar -xzf source.tar.gz\n"
        f"pip install -q -r cloud/requirements.txt\n"
        f"python cloud/train_torch_trm.py "
        f"--model-id {model_id} "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
        f"--iterations {iterations} "
        f"--reasoning-steps {reasoning_steps} "
        f"--decay-gamma {decay_gamma} "
        f"{'--act' if act else ''} "
        f"--gcs-output-bucket {output_bucket}\n"
    ]

    worker_pool = {
        "machineSpec": {
            "machineType": gpu_info["machine_type"]
        },
        "replicaCount": "1",
        "containerSpec": {
            "imageUri": DEFAULT_CONTAINER,
            "command": train_command
        }
    }

    if gpu_info["accelerator_count"] > 0:
        worker_pool["machineSpec"]["acceleratorType"] = gpu_info["accelerator_type"]
        worker_pool["machineSpec"]["acceleratorCount"] = gpu_info["accelerator_count"]

    spec = {
        "displayName": display_name,
        "jobSpec": {
            "workerPoolSpecs": [worker_pool]
        }
    }
    return spec

def package_and_upload_source(
    bucket_name: str,
    project_id: str
) -> str:
    """Packages cloud/ and data/ directories into a tarball and uploads to GCS."""
    timestamp = int(time.time())
    archive_name = f"trm_source_{timestamp}.tar.gz"
    gcs_uri = f"gs://{bucket_name}/source/{archive_name}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, archive_name)
        with tarfile.open(tar_path, "w:gz") as tar:
            for folder in ["cloud", "data"]:
                if os.path.exists(folder):
                    tar.add(folder, arcname=folder)
        
        print(f"[Vertex Dispatcher] Uploading {tar_path} -> {gcs_uri} ...")
        cmd = ["gcloud", "storage", "cp", tar_path, gcs_uri, f"--project={project_id}"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to upload source archive: {res.stderr}")
            
    return gcs_uri

def submit_vertex_job(
    project_id: str,
    region: str,
    job_spec: Dict[str, Any],
    dry_run: bool = False
) -> Dict[str, Any]:
    display_name = job_spec.get("displayName", f"trm-gemma-train-{int(time.time())}")
    # gcloud ai custom-jobs create --config expects CustomJobSpec (the contents of jobSpec)
    custom_job_spec = job_spec.get("jobSpec", job_spec)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(custom_job_spec, f, indent=2)
        spec_file = f.name

    try:
        if dry_run:
            print("[Vertex Dispatcher] Dry-run enabled. Job specification payload:")
            print(json.dumps(job_spec, indent=2))
            return {
                "status": "DRY_RUN_VALIDATED",
                "display_name": display_name,
                "project": project_id,
                "region": region,
                "console_url": f"https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}"
            }

        cmd = [
            "gcloud", "ai", "custom-jobs", "create",
            f"--project={project_id}",
            f"--region={region}",
            f"--display-name={display_name}",
            f"--config={spec_file}",
            "--format=json"
        ]
        print(f"[Vertex Dispatcher] Submitting custom job {display_name} to Vertex AI ({region})...")
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if res.returncode != 0:
            raise RuntimeError(f"gcloud ai custom-jobs create failed:\n{res.stderr}")
            
        data = json.loads(res.stdout) if res.stdout.strip() else {}
        job_id = data.get("name", "").split("/")[-1] if "name" in data else display_name
        console_url = f"https://console.cloud.google.com/vertex-ai/locations/{region}/training/{job_id}?project={project_id}"
        
        print("=" * 70)
        print("🎉 Vertex AI Custom Training Job Successfully Submitted!")
        print("=" * 70)
        print(f"Job ID:      {job_id}")
        print(f"Status:      PENDING / QUEUED")
        print(f"Console URL: {console_url}")
        print(f"Stream Logs: gcloud ai custom-jobs stream-logs {job_id} --project={project_id} --region={region}")
        print("=" * 70)
        
        return {
            "status": "SUBMITTED",
            "job_id": job_id,
            "console_url": console_url,
            "project": project_id,
            "region": region
        }
    finally:
        if os.path.exists(spec_file):
            os.remove(spec_file)

def main():
    parser = argparse.ArgumentParser(description="Submit Tiny Recursive Gemma training job to Google Vertex AI.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="GCP Project ID")
    parser.add_argument("--region", default=DEFAULT_REGION, help="GCP Region")
    parser.add_argument("--staging-bucket", default=DEFAULT_BUCKET, help="GCS staging bucket")
    parser.add_argument("--gpu", default="L4", choices=["L4", "T4", "CPU"], help="Accelerator hardware")
    parser.add_argument("--model-id", default=os.getenv("BASE_MODEL", "google/gemma-4-E2B-it-qat-q4_0-unquantized"), help="Gemma 4 model ID")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--iterations", type=int, default=3, help="Recurrent steps (T)")
    parser.add_argument("--reasoning-steps", type=int, default=2, help="Reasoning steps (n)")
    parser.add_argument("--decay-gamma", type=float, default=1.5, help="Deep supervision decay gamma")
    parser.add_argument("--act", action="store_true", default=True, help="Enable ACT halting")
    parser.add_argument("--dry-run", action="store_true", help="Print job spec without submitting")
    args = parser.parse_args()

    timestamp = int(time.time())
    display_name = f"trm-gemma-train-{timestamp}"

    if args.dry_run:
        gcs_source = f"gs://{args.staging_bucket}/source/dry_run_source.tar.gz"
    else:
        gcs_source = package_and_upload_source(args.staging_bucket, args.project)

    spec = generate_job_spec(
        display_name=display_name,
        gcs_source_uri=gcs_source,
        gpu_type=args.gpu,
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        iterations=args.iterations,
        reasoning_steps=args.reasoning_steps,
        decay_gamma=args.decay_gamma,
        act=args.act,
        output_bucket=args.staging_bucket
    )

    submit_vertex_job(args.project, args.region, spec, dry_run=args.dry_run)

if __name__ == "__main__":
    main()

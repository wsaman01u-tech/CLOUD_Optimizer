"""Deployment Agent.
Converts autonomous scaling decisions into validated Infrastructure-as-Code (Terraform)
and Kubernetes manifests with strict rate-limiting and boundary safety rules.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


class DeploymentAgent:
    """Manages cloud infrastructure automation and deployment safety."""

    def __init__(
        self,
        min_servers: int = 1,
        max_servers: int = 10,
        output_dir: str = "outputs/terraform",
        dry_run: bool = True,
    ):
        self.min_servers = min_servers
        self.max_servers = max_servers
        self.output_dir = output_dir
        self.dry_run = dry_run
        self.history = []

    def validate_action(
        self,
        servers_current: int,
        target_servers: int,
    ) -> Tuple[int, str]:
        """Enforce strict safety rules on target server count.

        Rules:
        1. Clamp between [min_servers, max_servers]
        2. Rate-limit changes: max +/- 1 server delta per step
        """
        # Rule 1: Boundary checks
        clamped = max(self.min_servers, min(target_servers, self.max_servers))

        # Rule 2: Rate limit
        delta = clamped - servers_current
        safe_servers = servers_current
        status_msg = "Safe"

        if delta > 1:
            safe_servers = servers_current + 1
            status_msg = f"Rate limited: requested +{delta}, throttled to +1"
        elif delta < -1:
            safe_servers = servers_current - 1
            status_msg = f"Rate limited: requested {delta}, throttled to -1"
        else:
            safe_servers = clamped
            if clamped != target_servers:
                status_msg = f"Boundary enforced: requested {target_servers}, clamped to {clamped}"

        return safe_servers, status_msg

    def generate_terraform(self, server_count: int) -> Dict[str, str]:
        """Generate Terraform variable configuration files."""
        os.makedirs(self.output_dir, exist_ok=True)

        variables_content = f"""# Autonomous Cloud Optimizer Terraform Configuration
# Generated automatically by Deployment Agent

variable "instance_count" {{
  description = "Target scaled replica count of worker VM instances"
  type        = number
  default     = {server_count}
}}

variable "instance_type" {{
  description = "Compute instance machine type"
  type        = string
  default     = "Standard_D2s_v3"
}}

variable "environment" {{
  description = "Deployment environment"
  type        = string
  default     = "production"
}}
"""

        tfvars_content = f"""# Terraform runtime input variables
instance_count = {server_count}
environment    = "production"
updated_at     = "{datetime.now().isoformat()}"
"""

        variables_path = os.path.join(self.output_dir, "variables.tf")
        tfvars_path = os.path.join(self.output_dir, "terraform.tfvars")

        with open(variables_path, "w") as f:
            f.write(variables_content)

        with open(tfvars_path, "w") as f:
            f.write(tfvars_content)

        return {
            "variables_path": variables_path,
            "tfvars_path": tfvars_path,
            "variables_content": variables_content,
            "tfvars_content": tfvars_content,
        }

    def generate_kubectl(self, server_count: int, deployment: str = "cloud-worker-deployment") -> str:
        """Generate Kubernetes scaling CLI command."""
        return f"kubectl scale deployment {deployment} --replicas={server_count}"

    def generate_k8s_manifest(self, server_count: int, deployment: str = "cloud-worker-deployment") -> str:
        """Generate Kubernetes Deployment YAML manifest."""
        manifest = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment}
  labels:
    app.kubernetes.io/managed-by: autonomous-cloud-optimizer
spec:
  replicas: {server_count}
  selector:
    matchLabels:
      app: cloud-worker
  template:
    metadata:
      labels:
        app: cloud-worker
    spec:
      containers:
      - name: worker
        image: cloud-worker:v1.0
        resources:
          limits:
            cpu: "1000m"
            memory: "2Gi"
          requests:
            cpu: "500m"
            memory: "1Gi"
"""
        manifest_path = os.path.join(self.output_dir, "deployment.yaml")
        with open(manifest_path, "w") as f:
            f.write(manifest)
        return manifest

    def execute_scaling(
        self,
        servers_current: int,
        target_servers: int,
        reason: str = "RL optimization",
    ) -> Dict[str, Any]:
        """Safely calculate and generate all deployment artifacts for the step."""
        safe_count, safety_status = self.validate_action(servers_current, target_servers)
        tf_artifacts = self.generate_terraform(safe_count)
        kubectl_cmd = self.generate_kubectl(safe_count)
        k8s_yaml = self.generate_k8s_manifest(safe_count)

        record = {
            "timestamp": datetime.now().isoformat(),
            "servers_before": servers_current,
            "target_requested": target_servers,
            "servers_deployed": safe_count,
            "safety_status": safety_status,
            "reason": reason,
            "dry_run": self.dry_run,
            "kubectl_command": kubectl_cmd,
            "terraform_path": tf_artifacts["tfvars_path"],
        }
        self.history.append(record)
        return record

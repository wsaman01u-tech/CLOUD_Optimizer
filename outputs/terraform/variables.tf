# Autonomous Cloud Optimizer Terraform Configuration
# Generated automatically by Deployment Agent

variable "instance_count" {
  description = "Target scaled replica count of worker VM instances"
  type        = number
  default     = 6
}

variable "instance_type" {
  description = "Compute instance machine type"
  type        = string
  default     = "Standard_D2s_v3"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

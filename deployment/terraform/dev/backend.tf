terraform {
  backend "gcs" {
    bucket = "qwiklabs-gcp-00-200b6d849712-terraform-state"
    prefix = "dev"
  }
}

# Add this anywhere in your app.py before the route
from io import BytesIO
import sys

import boto3

from project.exception.exception import ProjectException
from project.utils.main_utils.utils import load_object

class S3Helper:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self._cache = {}  # Simple caching mechanism

    def load_from_s3(self, bucket_name, key):
        try:
            # Check cache first
            if key in self._cache:
                return self._cache[key]
                
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            stream = BytesIO(response['Body'].read())
            obj = load_object(stream)  # Now passing BytesIO directly
            self._cache[key] = obj  # Cache the loaded object
            return obj
        except Exception as e:
            raise ProjectException(e, sys)
    
    def get_latest_model_key(self, bucket_name, prefix):
        response = self.s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter='/'
        )
        folders = [cp['Prefix'] for cp in response.get('CommonPrefixes', [])]
        if not folders:
            raise ValueError(f"No models found in S3 bucket {bucket_name} with prefix {prefix}")
        return sorted(folders, reverse=True)[0]

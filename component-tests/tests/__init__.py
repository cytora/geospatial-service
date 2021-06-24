import os

env = os.environ.get('ENV')
service = os.environ.get('SERVICE')
SERVER_URL = os.environ.get('SERVER_URL', f'https://{service}-test.{env}.com')
AUTH_KEY = os.environ.get('AUTH_KEY')

headers = {"Authorization": AUTH_KEY}
version = "v1"


def url(endpoint):
    return f"{SERVER_URL}/{version}/{endpoint}"

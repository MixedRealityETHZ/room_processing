from urllib.parse import urljoin

import requests


class VrraApi:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()

    def api_call(self, method, url, **kwargs):
        res = self.session.request(method, urljoin(self.base_url, url), **kwargs)
        res.raise_for_status()
        return res

    # Queue API
    def queue_pop(self):
        res = self.api_call("GET", "/queue/pop")
        if res.status_code == 204:
            return None
        return res.json()

    def queue_push(self, body):
        return self.api_call("POST", "/queue", json=body).json()

    def set_task_completed(self, id, body):
        self.api_call("POST", f"/queue/{id}/completed", json=body)

    def set_task_failed(self, id, body):
        self.api_call("POST", f"/queue/{id}/failed", json=body)

    # Asset API
    def add_asset(self, body):
        return self.api_call("POST", "/assets", json=body).json()

    def get_asset(self, id):
        return self.api_call("GET", f"/assets/{id}").json()

    def set_asset_uploaded(self, id):
        return self.api_call("POST", f"/assets/{id}/uploaded")

    def download(self, url, f, chunk_size=1024):
        res = self.session.get(url, stream=True)
        res.raise_for_status()
        for chunk in res.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
        return res

    def upload(self, url, f):
        res = self.session.put(url, data=f)
        res.raise_for_status()
        return res

    def create_asset(self, name, file):
        asset = self.add_asset({"name": name})
        self.upload(asset["url"], file)
        self.set_asset_uploaded(asset["id"])
        return asset

    # Model API
    def add_model(self, body):
        return self.api_call("POST", "/models", json=body).json()

    # Room API
    def add_room(self, body):
        return self.api_call("POST", "/rooms", json=body).json()

    def add_room_obj(self, id, body):
        return self.api_call("POST", f"/rooms/{id}/objects", json=body).json()

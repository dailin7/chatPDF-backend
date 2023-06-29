from typing import List
from ..models.models import Collection


def add_filenames(collection_name: str, filenames: List[str]) -> None:
    collection = Collection.objects.get(collection_name=collection_name)
    collection.files = filenames + collection.files
    collection.save()


def get_filenames(collection_name: str) -> List[str]:
    return Collection.objects.get(collection_name=collection_name).files

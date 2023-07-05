from typing import List
from ..models.models import Collection
from ..serializers import CollectionSerializer
from django.core.exceptions import ValidationError
from backend.sources import qdrant_client, embedding
from qdrant_client.models import Distance, VectorParams


def create_collection(data) -> None:
    serializer = CollectionSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
    else:
        raise ValidationError("Invalid data")

    # create collection also in qdrant db
    collection_name = data["collection_name"]
    try:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    except Exception as e:
        raise ValidationError("Unable to create collection in qdrant db")
    return serializer.data


def add_filenames(collection_name: str, filenames: List[str]) -> None:
    collection = Collection.objects.get(collection_name=collection_name)
    collection.files = filenames + collection.files
    collection.save()


def get_filenames(collection_name: str) -> List[str]:
    return Collection.objects.get(collection_name=collection_name).files

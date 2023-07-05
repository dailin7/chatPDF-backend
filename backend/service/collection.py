from typing import List
from ..models.models import Collection
from ..serializers import CollectionSerializer
from django.core.exceptions import ValidationError
from backend.sources import qdrant_client, embedding
from qdrant_client.models import Distance, VectorParams
from django.http.request import QueryDict


def create_collection(collection_name: str) -> None:
    collection, created = Collection.objects.update_or_create(
        collection_name=collection_name
    )
    # serializer = CollectionSerializer(data=c)
    # if serializer.is_valid():
    #     serializer.save()
    # else:
    #     raise ValidationError("Invalid data")

    # create collection also in qdrant db
    try:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    except Exception as e:
        raise ValidationError("Unable to create collection in qdrant db")
    return collection


def delete_collection(collection_name: str) -> None:
    # try:
    collections = Collection.objects.filter(collection_name=collection_name)
    if collections.count() != 0:
        collections.delete()
    else:
        Warning("Collection not found in db")
    # except Exception as e:
    #     raise ValidationError("Unable to delete collection in db")
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception as e:
        raise ValidationError("Unable to delete collection in qdrant db")


def add_filenames(collection_name: str, filenames: List[str]) -> None:
    collection = Collection.objects.get(collection_name=collection_name)
    collection.files = filenames + collection.files
    collection.save()


def get_filenames(collection_name: str) -> List[str]:
    return Collection.objects.get(collection_name=collection_name).files

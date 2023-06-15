from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from backend.sources import qdrant_client
from qdrant_client.models import Distance, VectorParams
from ..utils import load_files
from qdrant_client.http.exceptions import UnexpectedResponse


@api_view(["POST"])
def create_collection(request):
    try:
        collection_name = request.POST["collection_name"]
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=100, distance=Distance.COSINE),
        )
        return Response(status=status.HTTP_200_OK)
    except:
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def get_collections(request):
    try:
        collection_res = qdrant_client.get_collections()
        res = [x.name for x in collection_res.collections]
        return Response(res, status=status.HTTP_200_OK)
    except:
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def get_collection(request, collection_name: str):
    try:
        collection = qdrant_client.get_collection(collection_name=collection_name)
        return Response(collection, status=status.HTTP_200_OK)
    except UnexpectedResponse:
        return Response(
            {"message": f"Collection '{collection_name}' does not exists!"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except:
        return Response(status=status.HTTP_400_BAD_REQUEST)

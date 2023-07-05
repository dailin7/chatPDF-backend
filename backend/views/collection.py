from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from ..serializers import CollectionSerializer
from ..models.models import Collection
import uuid


@DeprecationWarning
@api_view(["POST"])
def create_collection(request):
    serializer = CollectionSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    else:
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def get_collection(request, collection_name: str):
    # TODO: get by id instead of name
    collection = Collection.objects.get(collection_name=collection_name)
    serializer = CollectionSerializer(collection)
    return Response(serializer.data, status=status.HTTP_200_OK)

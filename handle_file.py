import logging
import docx2txt
import fitz


def handle_file(file):
    # TODO: change to MIME format
    filename = file.split("/")[-1].split(".")[0]
    logging.info("[handle_file] Handling file: {}".format(filename))

    try:
        extracted_text = extract_text_from_file(file)
    except ValueError as e:
        logging.error("[handle_file] Error extracting text from file: {}".format(e))
        raise e
    return handle_file_string(filename, extracted_text)


def extract_text_from_file(file):
    """Return the text content of a file."""
    # TODO: change the type to MIME type

    # get extension of a file
    extension = file.split(".")[-1]
    extracted_text - ""

    if extension == "pdf":
        doc = fitz.open(file)
        for page in doc:
            extracted_text += page.get_text()

    elif extension == "txt":
        # read file based on path
        with open(file, "r") as f:
            extracted_text = f.read()
        f.close()

    elif extension == "docx":
        extracted_text = docx2txt.process(file)

    else:
        raise ValueError("File type not supported")

    return extracted_text


def handle_file_string(filename, file_body_string):
    """Handle a file string by creating embeddings and upserting them to qdrant."""
    logging.info("[handle_file_string] Starting...")

    # Clean up the file string by replacing newlines and double spaces
    clean_file_body_string = file_body_string.replace("\n", "; ").replace("  ", " ")

    text_to_embed = "Filename is: {}; {}".format(filename, clean_file_body_string)

    # Create embeddings for the text
    try:
        text_embeddings, average_embedding = create_embeddings_for_text(
            text_to_embed, tokenizer
        )

        logging.info("[handle_file_string] Created embedding for {}".format(filename))

    except Exception as e:
        logging.error("[handle_file_string] Error creating embedding: {}".format(e))
        raise e

    # Get the vectors array of triples: file_chunk_id, embedding, metadata for each embedding
    # Metadata is a dict with keys: filename, file_chunk_index

try:
    from chunk_documents import process_document
    import os

    # Process the sample document
    try:
        process_document(
            file_path='sample_philosophy.txt',
            document_id='sample_thomistic_essay',
            title='Sample Thomistic Essay on Essence and Existence',
            chunk_size=200,  # Small chunks for testing
            overlap=50
        )
        print('✓ Successfully chunked and stored sample document')
    except Exception as e:
        print(f'✗ Error processing document: {e}')
except Exception as e:
    print(f'✗ Import error: {e}')
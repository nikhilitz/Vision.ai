def process_blob(blob):
    if blob.path == b"model.pth":
        blob.oid = git_lfs_migrate_blob(blob)
    return blob
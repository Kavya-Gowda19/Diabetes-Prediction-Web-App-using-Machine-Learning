import secrets

# Generate a secret key
secret_key = secrets.token_hex(16)
print(f"Your generated secret key: {secret_key}")
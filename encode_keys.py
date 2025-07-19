import base64

with open("secret.key", "rb") as f:
    secret_key_b64 = base64.b64encode(f.read()).decode()

with open("encrypted_password.bin", "rb") as f:
    encrypted_password_b64 = base64.b64encode(f.read()).decode()

print("SECRET_KEY_BASE64=" + secret_key_b64)
print("ENCRYPTED_PASSWORD_BASE64=" + encrypted_password_b64)

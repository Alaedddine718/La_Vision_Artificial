# 📤 Guía para Subir el Proyecto a GitHub

Sigue estos pasos para subir tu proyecto a GitHub.

---

## 📋 Prerequisitos

1. Tener una cuenta de GitHub ([crear cuenta](https://github.com/join))
2. Tener Git instalado en tu computadora
   ```bash
   # Verificar instalación
   git --version
   ```

---

## 🚀 Pasos para Subir a GitHub

### 1️⃣ Crear un Repositorio en GitHub

1. Ve a [GitHub](https://github.com)
2. Haz clic en el botón **"New"** o **"+"** → **"New repository"**
3. Configura el repositorio:
   - **Repository name**: `cnn-cifar10` (o el nombre que prefieras)
   - **Description**: `Clasificador de imágenes CIFAR-10 con CNN y Streamlit`
   - **Visibilidad**: Public o Private (tu elección)
   - ⚠️ **NO** marques "Add a README file" (ya lo tenemos)
   - ⚠️ **NO** añadas .gitignore ni licencia (ya los tenemos)
4. Haz clic en **"Create repository"**

---

### 2️⃣ Inicializar Git Localmente

Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
# Navega a la carpeta del proyecto
cd /Users/alessiocicilano/Downloads/cnn-cifar10-main

# Inicializa el repositorio Git
git init

# Añade todos los archivos
git add .

# Verifica qué archivos se añadirán (opcional)
git status

# Crea el primer commit
git commit -m "Initial commit: Clasificador CNN CIFAR-10 con interfaz Streamlit"
```

---

### 3️⃣ Conectar con GitHub

Reemplaza `TU-USUARIO` con tu nombre de usuario de GitHub:

```bash
# Añade el repositorio remoto
git remote add origin https://github.com/TU-USUARIO/cnn-cifar10.git

# Verifica que se añadió correctamente
git remote -v

# Renombra la rama principal a 'main' (si es necesario)
git branch -M main
```

---

### 4️⃣ Subir el Código

```bash
# Sube los archivos a GitHub
git push -u origin main
```

Si te pide autenticación:
- **Método 1**: Usa un Personal Access Token (recomendado)
  - Ve a GitHub → Settings → Developer settings → Personal access tokens
  - Genera un nuevo token con permisos de `repo`
  - Usa el token como contraseña

- **Método 2**: Usa SSH (más avanzado)
  - Configura una clave SSH siguiendo [esta guía](https://docs.github.com/es/authentication/connecting-to-github-with-ssh)

---

## ✅ Verificación

1. Ve a tu repositorio en GitHub: `https://github.com/TU-USUARIO/cnn-cifar10`
2. Deberías ver todos tus archivos
3. El README.md se mostrará automáticamente en la página principal

---

## 🔄 Actualizaciones Futuras

Cuando hagas cambios en el código:

```bash
# Añade los archivos modificados
git add .

# Crea un commit con descripción
git commit -m "Descripción de los cambios"

# Sube a GitHub
git push
```

---

## 📝 Comandos Git Útiles

```bash
# Ver estado de los archivos
git status

# Ver historial de commits
git log --oneline

# Ver archivos que se ignorarán
git status --ignored

# Deshacer cambios no guardados
git checkout -- archivo.py

# Ver diferencias
git diff
```

---

## 🎯 Añadir Colaboradores

1. Ve a tu repositorio en GitHub
2. **Settings** → **Collaborators**
3. Haz clic en **"Add people"**
4. Busca por nombre de usuario o email

---

## 🏷️ Crear Releases (Opcional)

Para versionar tu proyecto:

1. Ve a tu repositorio en GitHub
2. **Releases** → **"Create a new release"**
3. Añade un tag (ej: `v1.0.0`)
4. Describe los cambios
5. Publica

---

## 📌 Consejos

✅ Haz commits frecuentes con mensajes descriptivos  
✅ No subas archivos grandes (modelos .h5 están en .gitignore)  
✅ Mantén el README actualizado  
✅ Usa branches para nuevas funcionalidades  
✅ Documenta cambios importantes en CHANGELOG.md (opcional)  

---

## 🆘 Problemas Comunes

**Error: "remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/TU-USUARIO/cnn-cifar10.git
```

**Error: "Permission denied"**
- Verifica tus credenciales
- Usa un Personal Access Token en lugar de contraseña

**Archivos grandes rechazados**
- Verifica que .gitignore esté configurado correctamente
- Los modelos .h5 no deberían subirse

---

¡Listo! Tu proyecto ya está en GitHub 🎉


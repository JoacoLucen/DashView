# Credenciales de acceso — DashView (DEMO)

> ⚠️ **Datos de prueba.** Estas credenciales son solo para uso interno del grupo
> durante el desarrollo y la demo. No representan información sensible ni real.
> Para un entorno productivo habría que mover las contraseñas a hashing
> (`werkzeug.security`) y a un almacén fuera del código.

## Cómo ingresar

1. Levantar la app: `python app.py` → abrir http://localhost:8050
2. En la pantalla de **login**, escribir usuario y contraseña.
3. Botón **Ingresar** (o tecla Enter).
4. Para cerrar sesión: botón **Salir** en la barra superior.

La sesión queda guardada en el navegador y se mantiene al recargar, hasta hacer
**logout** explícito.

## Usuarios

Cada usuario es un *stakeholder* y accede **únicamente** a su pestaña principal.

| Usuario     | Contraseña      | Stakeholder           | Pestaña visible          |
|-------------|-----------------|-----------------------|--------------------------|
| `marketing` | `marketing123`  | Marketing             | Marketing                |
| `direccion` | `direccion123`  | Dirección General     | Dirección General        |
| `retencion` | `retencion123`  | Retención             | Retención y Facturación  |
| `producto`  | `producto123`   | Equipo de Producto    | Equipo de Producto       |

> Si se ingresan credenciales incorrectas, aparece el mensaje
> *"Usuario o contraseña incorrectos."*

## Dónde se definen

La base de usuarios vive en el diccionario `USERS` dentro de `app.py`. Para
agregar, quitar o cambiar usuarios/contraseñas, editar ese diccionario.

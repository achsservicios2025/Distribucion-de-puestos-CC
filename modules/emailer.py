import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import os

def send_reservation_email(to_email, subject, body_html, logo_path="static/logo.png"):
    """
    Envía un correo HTML con el logo incrustado (si es posible).
    Asegura que el remitente sea el usuario de autenticación SMTP para cumplir con Brevo.
    """
    # Intentar obtener credenciales de secrets
    try:
        smtp_server = st.secrets["smtp"]["server"]
        smtp_port = st.secrets["smtp"]["port"]
        smtp_user = st.secrets["smtp"]["user"] # Usuario Brevo (e.g., 9bdaad001@smtp-brevo.com)
        smtp_password = st.secrets["smtp"]["password"]
    except KeyError:
        print("ERROR SMTP: No se encontraron todas las credenciales SMTP en secrets.toml")
        return False
    except Exception as e:
        print(f"Error al cargar secretos SMTP: {e}")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    
    # ----------------------------------------------------
    # CORRECCIÓN CLAVE: Asignamos un nombre visible al remitente
    # Esto usa el email de autenticación (smtp_user) para garantizar el envío
    display_name = "ACHS Servicios - Gestión"
    msg["From"] = f"{display_name} <{smtp_user}>"
    # ----------------------------------------------------
    
    msg["To"] = to_email

    # Diseño HTML Profesional
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
            .header {{ background-color: #00A04A; padding: 20px; text-align: center; color: white; }}
            .content {{ padding: 20px; }}
            .footer {{ background-color: #f9f9f9; padding: 15px; text-align: center; font-size: 12px; color: #888; }}
            h2 {{ margin-top: 0; }}
            ul {{ background: #f0f8ff; padding: 15px; border-radius: 5px; }}
            li {{ list-style: none; padding: 5px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>ACHS Servicios</h1>
                <p>Confirmación de Reserva</p>
            </div>
            <div class="content">
                {body_html}
            </div>
            <div class="footer">
                <p>Este es un mensaje automático, por favor no responder.</p>
                <p>© ACHS Servicios - Gestión de Espacios</p>
            </div>
        </div>
    </body>
    </html>
    """

    part = MIMEText(html_content, "html")
    msg.attach(part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            # Aseguramos que el envío use el usuario autenticado como remitente
            server.sendmail(smtp_user, to_email, msg.as_string()) 
        return True
    except Exception as e:
        print(f"Error enviando email: Falló la conexión SMTP o la autenticación. Detalle: {e}")
        return False

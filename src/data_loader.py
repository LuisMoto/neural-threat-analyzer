import pandas as pd
import numpy as np
from sklearn.utils import resample
from config import DATA_DIR, SQLI_DATA, ENRON_DATA


SAFE_SPANISH_SAMPLES = [
    "Hola equipo, adjunto el reporte de métricas de esta semana. Por favor denle una leída antes de nuestra reunión del martes.",
    "Buenos días, les comparto el documento actualizado con los cambios solicitados. Quedo pendiente de sus comentarios.",
    "Estimados, les informo que la reunión de mañana se pospone para el jueves a las 10am. Saludos.",
    "Compañeros, el sistema estará en mantenimiento el sábado de 2am a 6am. Por favor guarden su trabajo antes.",
    "Hola, adjunto la factura correspondiente al mes de enero. Cualquier duda, con gusto les atiendo.",
    "El reporte trimestral ya está disponible en la carpeta compartida. Por favor revisarlo antes del viernes.",
    "Buenos días, les recuerdo que mañana tenemos la presentación del proyecto ante el cliente. Nos vemos a las 9am.",
    "Hola equipo, les comparto el resumen de la junta de ayer. Quedo atento a cualquier comentario.",
    "La nueva versión del sistema ya fue desplegada en producción. Si encuentran algún error, repórtenlo al área de soporte.",
    "Estimados, se les recuerda enviar su reporte de actividades antes del viernes a las 5pm.",
    "El documento final está listo para revisión. Pueden acceder desde la carpeta del proyecto en el servidor.",
    "Buenos días, adjunto las minutas de la reunión de la semana pasada para su revisión y aprobación.",
    "Hola, les informo que el acceso al sistema se restableció. Ya pueden ingresar con normalidad.",
    "Compañeros, mañana es el último día para enviar sus evaluaciones de desempeño. No olviden completarlas.",
    "El servidor de pruebas estará disponible a partir del lunes. Les haré llegar las credenciales de acceso.",
    "Buenos días a todos, adjunto el calendario de vacaciones aprobado para el segundo semestre.",
    "Hola equipo, el módulo de reportes ya fue actualizado con los cambios solicitados en la reunión anterior.",
    "Les informo que la capacitación del nuevo sistema será el próximo martes de 10am a 1pm en la sala de juntas.",
    "Estimados, les recuerdo que el día de mañana no habrá servicio de comedor por mantenimiento.",
    "Hola, adjunto el contrato revisado con los comentarios del área legal. Favor de revisar antes de firmar.",
    "El código fue revisado y aprobado. Pueden hacer el merge a la rama principal cuando estén listos.",
    "Buenos días, les comparto el link de la presentación que usaremos en la conferencia del viernes.",
    "Hola equipo, el sprint de esta semana fue completado. El tablero de Jira ya está actualizado.",
    "Se les notifica que el proceso de nómina ya fue ejecutado. Los depósitos se verán reflejados mañana.",
    "Estimados, adjunto el plan de trabajo para el siguiente trimestre para su revisión y retroalimentación.",

    "Hola Nathalia, te comparto la versión final del código comentado para el Neural Threat Analyzer. Hay que darle una última revisión antes de la presentación del 1 de junio.",
    "Buenos días, subí al repositorio los últimos avances del entorno de RV para el Laboratorio de Óptica Avanzada. Quedo atento a sus comentarios.",
    "Profesor, adjunto el documento con la parametrización de la ruta de la Línea 2 del Trolebús correspondiente a la entrega de cálculo.",
    "Estimados, anexo mi currículum actualizado para continuar con el proceso de la vacante de Customer Management Intern en BAT.",
    "Buen día, solicito información sobre el estatus del folio ingresado en el sistema CESAC respecto al reporte en la zona de Jardín Balbuena.",
    "Adjunto el archivo PGN con la reconstrucción de la partida de ajedrez para analizar los errores en la apertura.",
    "Hola, les comparto el comprobante de inscripción y el pago para el campeonato de ajedrez de este fin de semana en CDMX.",
    "Te envío el archivo de Excel con la planeación financiera y logística de esta semana para revisarlo más tarde con Connie y Jacobo.",
    "Hola Ángel, te mando el track de la guitarra acústica para la mezcla. Mantuve la atmósfera oscura y alternativa como acordamos.",
    "Buenos días, adjunto el script en Python y los dashboards de Power BI correspondientes a la detección de anomalías en los logs corporativos.",
    "Hola a todos, les envío la ruta de los museos que visitaremos este fin de semana, empezando por el Soumaya y terminando en el Jumex.",
    "El reporte de la investigación sobre logística en Alemania de 1939 ya está completo en la carpeta compartida de Goodnotes.",

    "El PR ya fue aprobado. Puedes revisar los detalles del merge en https://github.com/empresa/proyecto-backend/pull/42",
    "Hola, la documentación de la API se ha migrado. El nuevo portal está disponible en https://docs.mi-empresa.com/api/v2",
    "Adjunto el enlace a la hoja de cálculo de Google con el presupuesto del Q3: https://docs.google.com/spreadsheets/d/1A2B3C/edit",
    "Los resultados de la encuesta de clima laboral ya están tabulados. Pueden verlos en www.notion.so/resultados-clima-2026",
    "Compañeros, la reunión de retrospectiva será por Zoom. Les dejo la liga para conectarse: https://zoom.us/j/9876543210",
    "El ticket sobre el bug de inicio de sesión fue actualizado. Seguimiento aquí: https://jira.empresa.net/browse/SEC-105",
    "Hola, dejé los mockups finales para la nueva interfaz en Figma. Revisen este link: https://www.figma.com/file/xyz123/interfaz",
    "La configuración del entorno de pruebas de Unity ya está documentada en el wiki: https://confluence.empresa.com/display/DEV/Unity+Setup",
    "Estimados, el despliegue en AWS fue exitoso. Los logs de CloudWatch se pueden consultar en https://us-east-1.console.aws.amazon.com/cloudwatch/",
    "Les comparto el artículo que discutimos sobre detección de anomalías en ciberseguridad: https://medium.com/towards-data-science/anomaly-detection",
    "El borrador del ensayo sobre lingüística está en Overleaf. Aquí tienen acceso de edición: https://www.overleaf.com/project/64a2b1c",
    "Hola equipo, el nuevo dataset para entrenar el modelo de NLP ya está en el bucket de S3: https://s3.console.aws.amazon.com/s3/buckets/data-nlp",
    "Pueden descargar la última versión del instalador desde nuestra página oficial: www.software-empresa.com/descargas",
    "La factura de los servicios de nube de este mes ya se generó. El PDF está en https://app.facturacion.com/ver/998877",
    "Por favor, revisen la guía de estilos de código antes de hacer el próximo commit: https://github.com/google/styleguide/blob/gh-pages/pyguide.md",
    "El entorno de desarrollo en Python fue actualizado. Consulten las nuevas dependencias en https://pypi.org/project/scikit-learn/",

    "Profesor, el repositorio con mi proyecto final de Deep Learning se encuentra alojado en https://github.com/luis/neural-threat-analyzer",
    "Nathalia, te paso el link del paper sobre redes neuronales convolucionales que vamos a citar: https://arxiv.org/abs/1234.5678",
    "La inscripción al torneo de FIDE ya quedó confirmada. La lista de jugadores está publicada en https://ratings.fide.com/tournament_list.phtml",
    "Les comparto mi perfil de Lichess para que analicemos las últimas partidas de práctica: https://lichess.org/@/usuario-ajedrez",
    "El PDF con el temario de Matemáticas Aplicadas para el próximo semestre ya se puede descargar de la página de la UNAM: https://www.fciencias.unam.mx/docencia/licenciatura",
    "Chicos, nos vemos el sábado en el Aroma Café. Les dejo la ubicación de Google Maps: https://maps.app.goo.gl/aroma-cafe-cdmx",
    "Encontré este plugin para Guitar Pro que nos puede servir para el arreglo barroco. Lo pueden bajar de https://www.guitar-pro.com/plugins",
    "El manual para exportar los objetos 3D a C# está en este tutorial de YouTube: https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "Para la comida del viernes hice reservación en El Cardenal San Ángel. El menú está aquí: https://restaurante-elcardenal.com/menu",
    "Jonathan, aquí está el link para reservar la pista de mini-golf el fin de semana: https://minigolf-cdmx.com/reservas",
    "La visita al Museo del Banco de México requiere registro previo. Háganlo en https://museobancodemexico.mx/visita/",
    "Compré los boletos para la exposición temporal en el Museo Jumex. Los códigos QR están en https://boletos.fundacionjumex.org/mis-boletos",
    "Te mando la referencia bibliográfica sobre el Reloj Astronómico de Praga para tu investigación: https://es.wikipedia.org/wiki/Reloj_Astronomico_de_Praga",
    "Confirmación de tu pedido en Sukiya. Sigue el estado de la entrega en https://delivery.sukiya.com.mx/order/445566",
    "El comprobante de domicilio para el trámite lo puedes descargar directamente del portal de CFE: https://app.cfe.mx/Aplicaciones/CCFE/Recibos/",

    "Alerta de seguridad: Detectamos un nuevo inicio de sesión en tu cuenta de Google. Revisa tu actividad en https://myaccount.google.com/security",
    "Hola, hemos actualizado nuestra política de privacidad en Patreon. Lee el documento completo en https://www.patreon.com/policy",
    "Tu pedido de MercadoLibre está en camino. Rastréalo utilizando este enlace: https://www.mercadolibre.com.mx/rastreo/123",
    "Tu factura de Telcel está lista para ser descargada. Ingresa a Mi Telcel: https://www.mitelcel.com/facturas",
    "Recordatorio: Tienes una cita programada en la clínica. Confirma tu asistencia en https://pacientes.clinica.com/confirmar",
    "El pago de tu tarjeta de crédito ha sido procesado con éxito. Descarga el comprobante en https://www.bbva.mx/comprobantes",
    "Te informamos que los términos de servicio de tu cuenta de Oracle han sido actualizados. Detalles en https://www.oracle.com/legal/terms.html",
    "GitHub: Tienes notificaciones no leídas en el repositorio. Revisa tu bandeja de entrada en https://github.com/notifications",
    "Tu suscripción a Spotify Premium se renovará el próximo mes. Administra tu plan en https://www.spotify.com/account",
    "Se ha generado un nuevo recibo de nómina en el portal del empleado. Accede a https://nomina.empresa.com/login",
    "AWS Notification: El uso de tu capa gratuita está al 85%. Verifica tu panel de facturación en https://console.aws.amazon.com/billing",
    "LinkedIn: Alguien ha visto tu perfil recientemente. Descubre quién fue en https://www.linkedin.com/me/profile-views/",
    "Tu vuelo con Aeroméxico ha sido confirmado. Puedes hacer check-in 24 horas antes en https://aeromexico.com/checkin",
    "El reporte mensual de Google Analytics de tu sitio web ya está disponible: https://analytics.google.com/report/123",

    "El equipo de soporte resolvió tu solicitud. Por favor, califica nuestro servicio en https://support.empresa.com/feedback",
    "Adjunto el enlace del artículo sobre detección de errores en tableros de ajedrez mediante IA: https://arxiv.org/pdf/2104.1234.pdf",
    "Hola, subí las fotos de la visita al Museo Soumaya a una carpeta de Drive. Aquí está el link: https://drive.google.com/drive/folders/fotos-soumaya",
    "La junta de planeación financiera se agendó para mañana. El enlace de Google Meet es https://meet.google.com/abc-defg-hij",
    "Por favor revisen este foro de StackOverflow, tiene la solución al error de importación en Python: https://stackoverflow.com/questions/12345/import-error",
    "Te comparto el repositorio de PayloadsAllTheThings para usarlo de referencia en la clase de ciberseguridad: https://github.com/swisskyrepo/PayloadsAllTheThings",
    "El tutorial para limpiar la voz de las canciones usando IA está muy bueno, te lo dejo: https://www.youtube.com/watch?v=tutorial-audio",
    "Aquí pueden consultar la documentación de Power BI sobre detección de anomalías: https://learn.microsoft.com/es-es/power-bi/transform-model/desktop-anomaly-detection",
    "Te envío la liga de Amazon con la laptop Asus Zenbook 14 OLED que estábamos revisando: https://www.amazon.com.mx/dp/B00XYZ123",
    "El proceso de alta de proveedores cambió. Las nuevas políticas están descritas en https://intranet.empresa.com/proveedores",
    "El equipo de recursos humanos habilitó el portal para la solicitud de vacaciones: https://rrhh.empresa.com/vacaciones",
    "Les comparto la invitación al webinar sobre ingeniería de inteligencia artificial: https://www.eventbrite.com/e/ai-engineering-webinar",
    "La respuesta a tu queja sobre obstrucción vial en CDMX ha sido actualizada en el portal: https://tramites.cdmx.gob.mx/cesac/",
    "Descarga la guía completa para usar la API de Keras en NLP desde la página de TensorFlow: https://www.tensorflow.org/text/guide/word_embeddings",
    "La reunión mensual del club de ajedrez será en la cafetería. Votemos la hora aquí: https://doodle.com/poll/xyz123"
] * 4


def load_and_merge_data():
    print("\n--- Starting data merge ---")

    print("--- Loading safe emails (Ling + Enron Ham + Spanish synthetic) ---")

    ling_df = pd.read_csv(DATA_DIR / "Ling.csv")
    ling_df = ling_df[ling_df['label'] == 0][['body']].rename(columns={'body': 'Text'})

    enron_df = pd.read_csv(DATA_DIR / "enron_spam_data.csv")
    enron_df['Text'] = enron_df['Subject'].fillna('') + " " + enron_df['Message'].fillna('')
    enron_df = enron_df[enron_df['Spam/Ham'] == 'ham'][['Text']]

    spanish_safe_df = pd.DataFrame({'Text': SAFE_SPANISH_SAMPLES})

    safe_df = pd.concat([ling_df, enron_df, spanish_safe_df], ignore_index=True)
    safe_df['Target'] = 0

    print("--- Loading Phishing attacks ---")

    nazario  = pd.read_csv(DATA_DIR / "Nazario.csv")[['body']].rename(columns={'body': 'Text'})
    nigerian = pd.read_csv(DATA_DIR / "Nigerian_Fraud.csv")[['body']].rename(columns={'body': 'Text'})
    ceas     = pd.read_csv(DATA_DIR / "CEAS_08.csv")
    ceas     = ceas[ceas['label'] == 1][['body']].rename(columns={'body': 'Text'})

    phishing_df = pd.concat([nazario, nigerian, ceas], ignore_index=True)
    phishing_df['Target'] = 1


    print("--- Loading SQLi attacks ---")

    sqli_df = pd.read_csv(SQLI_DATA, on_bad_lines='skip')
    sqli_df['Label'] = pd.to_numeric(sqli_df['Label'], errors='coerce')
    sqli_df = sqli_df[sqli_df['Label'] == 1]
    sqli_df = sqli_df[['Sentence']].rename(columns={'Sentence': 'Text'})
    sqli_df['Target'] = 2

   
    df_raw = pd.concat([safe_df, phishing_df, sqli_df], ignore_index=True)
    df_raw = df_raw.dropna(subset=['Text'])
    df_raw = df_raw[df_raw['Text'].str.strip() != '']

    print("\n--- Raw class distribution (before balancing) ---")
    print(df_raw['Target'].value_counts())


    counts    = df_raw['Target'].value_counts()
    min_count = counts.min()
    target_n  = min(min_count, 15_000)

    balanced_parts = []
    for cls in df_raw['Target'].unique():
        cls_df = df_raw[df_raw['Target'] == cls]
        if len(cls_df) > target_n:
            cls_df = resample(cls_df, n_samples=target_n, replace=False, random_state=42)
        balanced_parts.append(cls_df)

    df_balanced = pd.concat(balanced_parts, ignore_index=True)

    print("\n--- Balanced class distribution ---")
    print(df_balanced['Target'].value_counts())

    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_merge_data()
    print(df['Target'].value_counts())
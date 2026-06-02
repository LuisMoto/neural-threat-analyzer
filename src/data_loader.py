# Importación de librerías para el manejo y balanceo de datos
import pandas as pd
import numpy as np
from sklearn.utils import resample
from config import DATA_DIR, SQLI_DATA, ENRON_DATA


# MUESTRAS SINTÉTICAS EN ESPAÑOL

# Los datasets públicos clásicos (Enron, Nazario, etc.) están en inglés. Para que nuestro modelo detecte amenazas en nuestro contexto real (universidad, empresas locales), decidí inyectar ejemplos creados manualmente en español.
# Esto previene que la red asocie directamente "idioma español = anomalía" y le enseña a identificar la intención semántica sin importar el idioma.

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

# Ejemplos manuales para enseñar la intención de un ciberataque en español
PHISHING_SPANISH_SAMPLES = [
    "Aviso importante: Tu cuenta de BBVA ha sido restringida temporalmente. Para reactivar tus servicios, ingresa aquí: http://bbva-alertas.xyz/reactivacion",
    "Estimado cliente, detectamos un cargo retenido en su tarjeta de crédito. Si no reconoce este cargo, cancele la operación en: http://192.168.1.15/seguridad/banamex",
    "Tu paquete de Estafeta no pudo ser entregado por falta de información. Actualiza tu dirección de entrega hoy mismo: http://estafeta-entregas.top/rastreo",
    "Soporte IT: Tu contraseña de correo corporativo caduca en 2 horas. Consérvala iniciando sesión en el nuevo portal: http://webmail-update.tk/login",
    "SAT: Notificación de adeudo fiscal. Descargue el comprobante de su multa para evitar recargos en el siguiente enlace: http://sat-gob.xyz/multas",
    "Netflix: No pudimos procesar tu último pago. Tu cuenta será suspendida al finalizar el día. Actualiza tu tarjeta en: http://netflix-pagos.online/renovar",
    "Amazon: Tu pedido #445-9823 ha sido retenido en aduana. Paga los gastos de importación para liberar el envío: http://amazon-aduanas.click/pago",
    "Felicidades, tu número fue seleccionado en el sorteo de aniversario de Telcel. Reclama tu equipo gratis aquí: http://premios-telcel.ml/ganador",
    "Aviso de Recursos Humanos: Descarga tu recibo de nómina con el ajuste salarial de este mes en el portal externo: http://nomina-empleados.top/descarga",
    "Microsoft 365: Se detectó un inicio de sesión inusual desde Rusia. Si no fuiste tú, protege tu cuenta inmediatamente: http://104.23.1.5/microsoft/secure",
    "CFE: Tu último recibo de luz presenta un saldo vencido. Evita el corte del servicio pagando en línea ahora: http://cfe-pagos.xyz/recibo",
    "MercadoLibre: Tu cuenta fue suspendida por actividad irregular. Para verificar tu identidad y recuperar el acceso, entra a: http://mercadolibre-seguro.tk/verificar",
    "Santander: Tienes una transferencia retenida por $15,000 MXN. Autoriza o cancela el movimiento en tu banca electrónica: http://santander-movimientos.top/auth",
    "Spotify: Tu suscripción Premium ha expirado. Renueva hoy con un 50% de descuento a través de este enlace exclusivo: http://spotify-promos.online/premium",
    "Administrador de red: El espacio de tu buzón está lleno. Amplía tu cuota de almacenamiento gratis ingresando a: http://quota-update.xyz/login",
    "Citibanamex: Su token móvil ha sido desincronizado. Para poder seguir haciendo transferencias, sincronícelo en: http://banamex-token.ml/sync",
    "DHL Express: El mensajero no encontró a nadie en el domicilio. Reprograme su entrega pagando la tarifa de reenvío: http://dhl-mexico.click/reprogramar",
    "Alerta de seguridad de Google: Alguien intentó acceder a tu cuenta. Cambia tu contraseña inmediatamente en: http://google-security.tk/update",
    "Hola, te comparto el documento escaneado que me pediste. Ábrelo directamente desde mi OneDrive personal: http://onedrive-compartido.xyz/documento.pdf",
    "Liverpool: Tienes un saldo a favor en tu monedero electrónico a punto de vencer. Úsalo hoy ingresando a: http://liverpool-recompensas.top/monedero",
    "Estimado proveedor, adjuntamos la orden de compra de este mes. Por favor confirme la recepción en nuestro portal: http://portal-proveedores.online/login",
    "Uber: Tu cuenta tiene un adeudo pendiente por tu último viaje. Liquida el saldo para poder seguir usando la app: http://uber-pagos.xyz/liquidar",
    "Aviso URGENTE: Su equipo está infectado con 3 virus. Descargue nuestro antivirus gratuito para limpiar su sistema: http://antivirus-scan.tk/download",
    "PayPal: Hemos limitado su cuenta temporalmente para protegerlo. Proporcione los datos solicitados para restaurarla: http://paypal-resolucion.ml/centro",
    "Telmex: Tu factura digital ya está disponible. Tienes un cargo adicional por servicios no reconocidos. Revísalo en: http://telmex-factura.top/detalle",
    "Hola, mira estas fotos del fin de semana, creo que sales en una de ellas. Descárgalas antes de que las borre: http://galeria-fotos.xyz/ver",
    "Apple ID: Su cuenta ha sido bloqueada por razones de seguridad. Desbloquéela verificando sus preguntas de seguridad en: http://apple-soporte.click/unlock",
    "Buro de Crédito: Tienes un nuevo reporte negativo en tu historial. Consulta el detalle y quién lo emitió en: http://buro-alertas.online/reporte",
    "Estimado estudiante, su inscripción al próximo semestre está detenida por falta de pago. Regularice su situación en: http://servicios-escolares.xyz/pago",
    "Zoom: Te han invitado a una reunión de carácter urgente. Únete a la sala de conferencias haciendo clic aquí: http://zoom-videocall.tk/join",

    "Hola, soy el director. Estoy en una junta y no puedo hablar. Necesito que compres 10 tarjetas de regalo de Apple de $1000 y me pases los códigos por aquí urgente.",
    "Compañeros, el área de sistemas está actualizando la base de datos. Por favor, respondan a este correo con su usuario y contraseña actual para no perder el acceso.",
    "Estimado cliente, nuestra cuenta bancaria principal está en mantenimiento. A partir de hoy, por favor realice todos los pagos de facturas a la nueva cuenta CLABE que le adjunto.",
    "Hola equipo, necesito que alguien de finanzas me apoye haciendo una transferencia urgente a un nuevo proveedor en el extranjero. Es confidencial, avísenme quién está disponible.",
    "Recursos Humanos: Estamos actualizando los expedientes. Por favor, envía copia de tu INE, comprobante de domicilio y estado de cuenta bancario respondiendo a este mensaje.",
    "Aviso importante: Si no confirmas tu asistencia a la capacitación respondiendo con tu número de empleado y NIP, se te descontará el día de mañana.",
    "Hola, perdí mi celular y estoy usando un correo prestado. Tuve una emergencia médica, ¿crees que me puedas prestar $2000? Te los deposito el lunes sin falta.",
    "Proveedor, detectamos un error en su última factura. Adjunto el PDF con los detalles. Si no lo corrigen para hoy a las 5pm, cancelaremos el contrato.",
    "Soporte Técnico: Hemos detectado actividad sospechosa en tu equipo. Responde a este correo con tu clave de administrador local para que podamos escanearlo remotamente.",
    "Hola, soy de contabilidad. Me rebotó el último pago de tu nómina. Pásame tu número de cuenta completo y la CLABE interbancaria para volver a intentarlo.",
    "Estimado usuario, su cuenta de correo será eliminada en 24 horas si no valida su identidad. Responda a este mensaje con la palabra 'CONFIRMAR' y su contraseña.",
    "Atención: Somos del equipo legal de la empresa. Necesitamos que nos envíes toda la información confidencial del proyecto actual para una auditoría sorpresa.",
    "Hola, estoy de viaje de negocios y mi tarjeta corporativa no pasa. ¿Puedes transferir $5000 a la cuenta del hotel? El lunes a primera hora te lo reembolso.",
    "Estimado, adjunto la cotización que solicitó. El archivo está protegido, la contraseña para abrir el PDF es su misma clave de acceso al sistema.",
    "Notificación de cobranza: Su cuenta tiene un atraso de 90 días. Adjuntamos la demanda mercantil. Responda este correo de inmediato para llegar a un acuerdo extrajudicial.",
    "Hola, ¿estás en la oficina? Necesito que me hagas un favor muy rápido, es completamente confidencial y no puedo decírselo a nadie más del equipo.",
    "Aviso de Sistemas: La VPN cambiará de configuración esta noche. Envíe sus credenciales actuales por este medio para que le generemos el nuevo perfil de conexión.",
    "Atención a todos: Por disposición oficial, necesitamos que confirmen su número de seguro social y RFC respondiendo a esta cadena antes del mediodía.",
    "Soy el proveedor de papelería. Cambiamos de razón social, por favor actualicen nuestra información en su sistema de pagos con la nueva cuenta que viene en el documento anexo.",
    "Hola, vi tu currículum en línea. Tenemos una vacante perfecta para ti. Para iniciar el proceso, envíanos un depósito de $500 para cubrir los gastos de tu examen médico.",
    "Estimado contribuyente, su declaración anual presenta inconsistencias. Responda este correo detallando sus ingresos de los últimos tres meses para evitar una auditoría.",
    "Equipo, el servidor de archivos se cayó. Quien tenga una copia local de la base de datos de clientes, por favor envíemela directamente a este correo personal.",
    "Hola, soy tu jefe. Olvidé mi computadora en la casa. Pásame tu usuario y contraseña de SAP rápido para autorizar unas órdenes de compra desde mi celular.",
    "Notificación urgente: Su póliza de seguro de auto ha sido cancelada por falta de pago. Envíe los datos de una nueva tarjeta de crédito para reactivarla de inmediato.",
    "Estimado socio, el pago de dividendos de este trimestre está listo. Confirme su número de cuenta bancaria y el nombre del titular para proceder con la transferencia.",
    "Atención: Detectamos un virus en la red local. Todos los empleados deben enviar su clave de acceso a sistemas@soporte-externo.com para verificar que no estén comprometidos.",
    "Hola, te escribo desde la gerencia. Necesito que modifiques la cuenta de depósito del proveedor principal y pongas esta nueva CLABE temporal para el pago de mañana.",
    "Aviso de nómina: Tienes un bono pendiente por cobrar. Para liberarlo, es necesario que nos confirmes tu salario actual y número de cuenta respondiendo a este correo.",
    "Estimado usuario, su paquete está retenido. Para agilizar la liberación, responda este mensaje adjuntando una foto de su tarjeta de crédito por ambos lados y su identificación.",
    "Hola, necesito acceso de administrador a la base de datos de producción por 5 minutos para arreglar un error crítico. Pásame las credenciales por aquí, yo me hago responsable."
] * 4


def load_and_merge_data():
    # Un reto crítico en ciberseguridad es la escasez de datasets multiclase que estén balanceados, para resolver esto, abordamos el problema sintetizando un corpus unificado a partir de cinco fuentes distintas
    print("\n--- Starting data merge ---")

    print("--- Loading safe emails (Ling + Enron Ham + Spanish synthetic) ---")
    
    # 1. CLASE 0: CONTENIDO SEGURO (SAFE) 
    # Para la clase de contenido seguro (0), usamos Enron Corpus y Ling-Spam para proveer una línea base de comunicación corporativa y profesional
    ling_df = pd.read_csv(DATA_DIR / "Ling.csv")
    ling_df = ling_df[ling_df['label'] == 0][['body']].rename(columns={'body': 'Text'})

    enron_df = pd.read_csv(DATA_DIR / "enron_spam_data.csv")
    enron_df['Text'] = enron_df['Subject'].fillna('') + " " + enron_df['Message'].fillna('')
    enron_df = enron_df[enron_df['Spam/Ham'] == 'ham'][['Text']]

    # Concatenamos nuestros datos sintéticos para dar robustez en nuestro entorno real
    spanish_safe_df = pd.DataFrame({'Text': SAFE_SPANISH_SAMPLES})

    safe_df = pd.concat([ling_df, enron_df, spanish_safe_df], ignore_index=True)
    safe_df['Target'] = 0

    print("--- Loading Phishing attacks ---")

    # 2. CLASE 1: ATAQUES DE PHISHING
    # Para la clase de Phishing (1), usamos Nazario, Nigerian Fraud y CEAS_08 para capturar tácticas de ingeniería social y marcadores de urgencia
    nazario  = pd.read_csv(DATA_DIR / "Nazario.csv")[['body']].rename(columns={'body': 'Text'})
    nigerian = pd.read_csv(DATA_DIR / "Nigerian_Fraud.csv")[['body']].rename(columns={'body': 'Text'})
    ceas     = pd.read_csv(DATA_DIR / "CEAS_08.csv")
    ceas     = ceas[ceas['label'] == 1][['body']].rename(columns={'body': 'Text'})
    
    phishing_synthetic = pd.DataFrame({'Text': PHISHING_SPANISH_SAMPLES})

    phishing_df = pd.concat([nazario, nigerian, ceas, phishing_synthetic], ignore_index=True)
    phishing_df['Target'] = 1

    print("--- Loading SQLi attacks ---")

    # 3. CLASE 2: INYECCIÓN SQL (SQLi) 
    # Para la clase de inyección SQL (2), usamos el SQLIV Technical Dataset para representar cargas útiles maliciosas estructuradas y la sintaxis de los exploits
    sqli_df = pd.read_csv(SQLI_DATA, on_bad_lines='skip')
    sqli_df['Label'] = pd.to_numeric(sqli_df['Label'], errors='coerce')
    sqli_df = sqli_df[sqli_df['Label'] == 1]
    sqli_df = sqli_df[['Sentence']].rename(columns={'Sentence': 'Text'})
    sqli_df['Target'] = 2
    
    # 4. LIMPIEZA INICIAL DE NULOS
    df_raw = pd.concat([safe_df, phishing_df, sqli_df], ignore_index=True)
    df_raw = df_raw.dropna(subset=['Text'])
    df_raw = df_raw[df_raw['Text'].str.strip() != '']

    print("\n--- Raw class distribution (before balancing) ---")
    print(df_raw['Target'].value_counts())


    # 5. BALANCEO ESTRATÉGICO DE DATOS (UNDERSAMPLING)

    # Calculamos el tamaño de la clase con menos muestras para emparejar hacia abajo (undersampling) limitando a un máximo de 15,000 muestras por clase para no saturar la RAM durante el entrenamiento.
    counts    = df_raw['Target'].value_counts()
    min_count = counts.min()
    target_n  = min(min_count, 15_000)

    balanced_parts = []
    for cls in df_raw['Target'].unique():
        cls_df = df_raw[df_raw['Target'] == cls]
        if len(cls_df) > target_n:
            # Para prevenir que el modelo se desvíe hacia la clase mayoritaria de contenido seguro, aplicamos una estricta estrategia de submuestreo (undersampling) para obtener una distribución balanceada
            # Esto asegura que la sensibilidad del modelo a las amenazas no se diluya por el alto volumen de datos legítimos
            cls_df = resample(cls_df, n_samples=target_n, replace=False, random_state=42)
        balanced_parts.append(cls_df)

    df_balanced = pd.concat(balanced_parts, ignore_index=True)

    print("\n--- Balanced class distribution ---")
    print(df_balanced['Target'].value_counts())

    # Finalmente, hacemos un shuffle (frac=1) para garantizar que los mini-batches de la red neuronal reciban muestras mezcladas durante el entrenamiento.
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_merge_data()
    print(df['Target'].value_counts())

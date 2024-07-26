import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def send_alert(subject, body):
    # Replace with your email configuration
    sender_email = "your_email@example.com"
    receiver_email = "alert_recipient@example.com"
    password = os.environ.get("EMAIL_PASSWORD")  # Set this as an environment variable

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

# Setup loggers
main_logger = setup_logger('main_logger', 'logs/main.log')
training_logger = setup_logger('training_logger', 'logs/training.log')
prediction_logger = setup_logger('prediction_logger', 'logs/prediction.log')
monitoring_logger = setup_logger('monitoring_logger', 'logs/monitoring.log')

# Example usage
if __name__ == "__main__":
    main_logger.info("This is a test log message")
    send_alert("Test Alert", "This is a test alert message")
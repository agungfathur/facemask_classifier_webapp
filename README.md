# facemask_webapp
Create simple webserver for facemask classification. Image data is sent in base64 format via HTTP post to webserver. <br />
You will get response value from webserver in 3 type of class

Using Flask as webserver classifier and ONNX runtime as classifier engine <br />
Tested and Run in Python 3.8.10

Installation

1. git clone https://github.com/agungfathur/facemask_webapp.git
2. cd facemask_webapp
3. Please install virtualenv for clean pip install library (recommendation) && dont forget to turn on the virtualenv
4. install dependencies --> **pip install -r requirements.txt**
5. run the webserver first --> **python main.py**
6. open new terminal, run the test program --> **python test.py** <br />
you can change the type of file in **test.py** by comment it (0_incorrect_mask.jpg, 1_using_mask.jpg, and 2_without_mask.jpg)

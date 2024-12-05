sudo yum update -y
sudo yum install python3 python3-pip -y
sudo yum install git -y

git clone https://github.com/adrielsumathipala/insurance-gpt.git
cd insurance-gpt

python3 -m venv .venv39
source .venv39/bin/activate
pip install -r requirements.txt

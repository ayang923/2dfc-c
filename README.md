# 2DFC-corners-C

Setup: 
```sudo apt install -y intel-oneapi-compiler-dpcpp-cpp```  
```wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null```  
```echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/intel-oneapi.list```  
```sudo apt update```  
```sudo apt install -y intel-oneapi-compiler-dpcpp-cpp```  
```source /opt/intel/oneapi/setvars.sh ```  
For the last step, the setvars.sh script might be in a different folder. The last command also only works for the current terminal session, so you can add to bashrc file.

Install intelmkl:
```sudo apt install intel-oneapi-mkl-devel```

yum -y update
yum -y upgrade

cd ~
yum install clang -y
yum install libpng-devel libtiff-devel zlib-devel libwebp-devel libjpeg-turbo-devel -y
yum install wget -y
wget https://github.com/DanBloomberg/leptonica/releases/download/1.80.0/leptonica-1.80.0.tar.gz
tar -xzvf leptonica-1.80.0.tar.gz
cd leptonica-1.80.0
./configure && make && make install

cd ~
wget https://mirror.squ.edu.om/gnu/autoconf-archive/autoconf-archive-2019.01.06.tar.xz
tar -xvf autoconf-archive-2019.01.06.tar.xz
cd autoconf-archive-2019.01.06
./configure && make && make install
cp m4/* /usr/share/aclocal/

cd ~
yum install libtool pkgconfig -y
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
mkdir tesseract
cd tesseract
wget https://github.com/tesseract-ocr/tesseract/archive/4.1.1.tar.gz
tar -zxvf 4.1.1.tar.gz
cd tesseract-4.1.1
./autogen.sh
./configure
make
make install

cd ~
mkdir tesseract-aws
cd tesseract-aws
cp /usr/local/bin/tesseract .
mkdir lib
cp /usr/local/lib/libtesseract.so.4 lib/
cp /usr/local/lib/liblept.so.5 lib/
cp /usr/lib64/libjpeg.so.62 lib/
cp /usr/lib64/libwebp.so.4 lib/
cp /usr/lib64/libstdc++.so.6 lib/
cp -r ~/tfenv/lib/python2.7/site-packages/* .
cp -r ~/tfenv/lib64/python2.7/site-packages/* .
mkdir tessdata
cd tessdata
wget https://github.com/tesseract-ocr/tessdata_fast/raw/master/osd.traineddata
wget https://github.com/tesseract-ocr/tessdata_fast/raw/master/eng.traineddata
mkdir configs
cp /usr/local/share/tessdata/configs/pdf configs/
cp /usr/local/share/tessdata/pdf.ttf .

cp -r /root/tesseract-aws/tessdata/* /usr/local/share/tessdata
set TESSDATA_PREFIX="/usr/local/share/tessdata"
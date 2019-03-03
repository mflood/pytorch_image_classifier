
echo "extracting images to jpg folder"
tar xzvf 102flowers.tgz

echo "making test / train / valid directory structures"
./mkdirs.sh

echo "Copying test images"
./cp_test.sh

echo "Copying train images"
./cp_train.sh

echo "Copying valid images"
./cp_valid.sh

echo "done"

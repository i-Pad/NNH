echo "********** WARNING **********"
echo "All directories in \"./result/AlexNet/\" will be deleted."
read -p "Do you wanna continue?(Y/n) " cond

while :
do
	if [ $cond = "Y" ]
	then
		rm -rf ./result/AlexNet/
		mkdir ./result/AlexNet

		for((i=0; i<5; i++));
		do
			mkdir ./result/AlexNet/conv$i
		done

		cp ./data/conv0before.h5 ./result/AlexNet/conv0/.

		for((i=0; i<5; i++));
		do
			python3 run.py <<< $i
		done

		break
	fi

	if [ $cond = "n" ]
	then
		break
	fi

	read -p "Do you wanna continue?(Y/n) " cond
done

test_data_path=./test_examples/AbdomenAtlasTest/
save_prediction_path=./test_examples/AbdomenAtlasPredict/

torchrun --master_port=21472 Atlas_test.py \
    --test_data_path $test_data_path --save_prediction_path $save_prediction_path
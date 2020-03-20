for i in {1..100..1}; do
    python main.py --train --ds_name transition --num_classes 2 \
       --network stn3d --out_dir ./trained/stn3d/trans_notrans_$i
done

#python main.py --train --ds_name transition --num_classes 2 \
#    --network stn3d  --out_dir ./trained/trans_notrans
#
#python main.py --test --ds_name transition --num_classes 3 \
#    --network stn3d --out_dir ./trained/trans_notrans/trans_notrans_39
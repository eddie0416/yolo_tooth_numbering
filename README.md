# yolo_tooth_numbering
目前的操作：
render.py遍歷ply/底下的所有.ply先進行渲染 有顏色00OMSZGW_lower_label.png跟沒有顏色00OMSZGW_lower_neutral.png各一張
存放位置{
    yolo_numbering_dataset/camera_params：相機參數
    yolo_numbering_dataset/dataset/images：渲染打光無label圖
    yolo_numbering_dataset/render_mask：渲染mask影像
}
接著creare_yolo_gt.py用來產生yolo標註框.txt(可以去讀取color_utils.py中的FDI2color來對應標註框的fdi class名稱)
遍歷yolo_numbering_dataset/render_mask中的所有_label.png，輸出標註框到yolo_numbering_dataset/dataset/labels
yologtvis.py則是可以將原圖跟標註框疊合進行視覺化
split.py進行資料切分
yolo中可以使用data.yaml進行class數字跟類別名稱的對應

後來修改了一下
目前如果要推論的話
要先使用prepare_data/pca_util.py先進行姿態校正，然後經過prepare_data/render.py渲染出純打光的影像(後來render.py有經過修改目前可以單純只渲染打光影像(推論時這樣做)，也可以輸出相機參數加上mask(訓練實這樣做))
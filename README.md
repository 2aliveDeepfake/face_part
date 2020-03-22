# face_part

사용법   

:얼굴 영역 전체 분할된 사진을 보고 싶을 때     
  python face_parts_detection.py -i "이미지 경로(상위폴더까지만)" -a true    
   ex. python face_parts_detection.py -i "images/" -a true

 :원하는 얼굴 영역 사진만 보고 싶을 때   
  python face_parts_detection.py -i "이미지 경로(상위폴더까지만)" -part "영역"   
	ex. python face_parts_detection.py -i "images/" -part "44,45,46,48,47"  
	영역 번호는 onenote 참조  

:원하는 포인트 대로 얼굴 영역을 지정해서 자르고 싶을 때   
영역이 하나일 경우  
python face_parts_detection.py -i "이미지 경로(상위폴더까지만)" -part "포인트번호,포인트번호,포인트번호"  
ex. python face_parts_detection.py -i "images/" -pts "69,17,36,37,38,39,21,71,70"  

영역이 여러개일 경우  
python face_parts_detection.py -i "이미지 경로(상위폴더까지만)" -part "포인트번호,포인트번호/포인트번호,포인트번호"  
ex. python face_parts_detection.py -i "images/" -pts "69,17,36,37,38,39,21,71,70/75,29,76,33"  

+ 필터이미지를 영역분할 하고 싶을 때 ** 필터이미지와 원본이미지의 이름이 같아야함 **
ex. python face_parts_detection.py -i 이미지 경로 -f 필터 이미지 경로 -pts 포인트번호
ex. python face_parts_detection.py -i D:/facebook/detector/blur/face_part_test/original/ -f D:/facebook/detector/blur/face_part_test/filter/ -pts 98,70,37,41,77,2,99/100,73,44,46,78,14,101

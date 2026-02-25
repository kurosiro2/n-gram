環境構築
・インストール方法はなんでも良いのでdockerをインストールしてください
<!-- Dockerfileがあるディレクトリでビルド  -->
docker build -t nsp-dev . 
<!-- コンテナ起動 -->
docker run -dit --name nsp-dev -v "$(pwd)":/work nsp-dev
<!-- 次回から -->
docker start nsp-dev
docker exec -it nsp-dev bash
・起動時にrootユーザで入ってしまうのを防ぎたい場合
例）docker exec -it --user $(id -u):$(id -g) nsp-dev bash



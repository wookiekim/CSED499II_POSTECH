# Outline of file exp whatevs

### Main
* set arguments
* configure directories
* trajs = LoadData()
  * inputdir 인 online-clusters directory에
  * cluster별로 저장되어있는 1.csv , 6.csv, 43.csv....
  * 결론적으로 `trajs[cluster][timestamp]`에 개수만큼 저장함
    * timestamp 는 datetime객체로, year, month, day, hour, minute, second까지 저장
    * `trajs`: dict, `trajs[cluster]`: sortedDict
* `top_cluster = pickle.load(f)
  * [documentation](https://docs.python.org/3/library/pickle.html)
  * `f = cluster path`, `cluster path = cluster-coverage/coverage.pickle`
* `Predict(args, config, top_cluster,trajs, method)`

### Predict `(args, config, top_cluster, trajs, method)`
* for date, cluster_list in top_cluster[args.start_pos // args.interval:- max(args.horizon // args.interval, 1)]:
  * top_cluster : list of tuples
  * tuple[0] :: datetime object
  * tuple[1] :: list of tuples
    * tuple[0] :: cluster number
    * tuple[1] :: would be number of clusters
  * args.start_pos :: 14400 at first
    * //= args.aggregate(==60 from run script) in `Main` **aggregate the results by how many minutes** == 14400/60
  * args.interval :: 1440 at first
    * //= args.aggregate(==60) in `Main` == 1440/60
  * args.horizon :: 4320  **how far we predict ahead** i
    * //= args.aggregate(==60) in `Main` == 72
  * `for date, cluster_list in top_cluster[10 : -3]:`
* train_delta_intervals = min(  (**calculated in minutes**)
  * ((date - first_date).days * 1440 + (date - first_date).seconds // 60) // (args.aggregate * args.interval), 
    * 10 * 1440 + ~0 // 60 * 1440/60  == 10
  * args.training_intervals) 
    * == 25
* `predict_delta_mins` = args.horizon * args.aggregate
  * == 72 * 60 == 4320
* ``` clusters = next(zip(*cluster_list))[:args.top_cluster_num] ```
    * `args.top_cluster_num` :: 3 **number of clusters to train together**
    * `cluster_list` is in descending order as such
      * `[(43, 3821152), (1, 505488), (6, 31412)]`
    * clusters = (43, 1, 6) :: just takes the cluster numbers only
* `data = GetMultiData(trajs, clusters, date, train_delta_intervals, args.interval, predict_delta_mins, args.aggregate)`

#### `GetMultiData(trajs, clusters, date, num_days, interval, num_mins, aggregate)`
* `date_list = [date - timedelta(minutes = x) for x in range(num_days * interval * aggregate, -num_mins, -aggregate)]`
  * `range(10 * 60 * 60, -4320, -60)`
    * 36000부터 -4320까지, -60씩
* `for date in date_list`
  * obs = []
  * for c in clusters: (43, 1 6)
    * if c in trajs: (load data에서 읽어왔다면~ 해당 csv가 존재한다면)
    *     `data_date = next(trajs[c].irange(maximum = date, inclusive = (True, False), reverse = True), None)`
    *      그저 해당 date보다 이전에 있던 (reverse = True라서 그런가...이유는 아직 잘 모름) date 가져오기
    *      `data_point = trajs[c][data_date]` 해당 date의 workload 값 
    *      `obs.append(data_point)`
    * traj.append(obs)
  * traj = np.array(traj)
* return traj ~~aggregate 별로 시계열대로 데이터를 받아온다는 점~~

#### ` data, data_min, data_mean, data_std = Normalize(data)`
* data :: final product
  * `data_min` :: 1 - np,min(data) || 0
  * `data_mean :: np.log(원래 data + data_min) 의 mean`
  * `data_std :: data -= data_mean 의 std`
* `고로 data :: normalized data (+ data_min, log, - own mean, / std)`

### Predict 이어서
* train_data = data[:-args.interval - args.horizon]
  * data[:-(96)]
* `test_data = data[-(args.paddling_intervals * args.interval + args.horizon + args.interval):]`
  * `args.paddling_intervals`:: 7
  * `args.interval` :: 24
  * `args.horizon` :: 72
  * data[-(264):] 
* 312 orig, 216 train, 264test
* model = GetModel(args, train_data, method)
  * return corresponding model (Transformer_Model.make_model?)
* criterion = nn.MSELoss() // which is KVLDLoss or sth in harvardnlp


    sudo apt install ros-galactic-navigation2 ros-galactic-nav2-bringup '~ros-galactic-turtlebot3-.*'

```
sudo apt-key adv --fetch-key https://repo.arrayfire.com/GPG-PUB-KEY-ARRAYFIRE-2020.PUB
echo "deb [arch=amd64] https://repo.arrayfire.com/debian all main" | sudo tee /etc/apt/sources.list.d/arrayfire.list

sudo apt update

sudo apt install arrayfire-cpu3-mkl
sudo apt install arrayfire-cpu3-dev
```

```
The following NEW packages will be installed:
  arrayfire-cpu3-mkl intel-comp-l-all-vars-19.1.0-166 intel-comp-nomcu-vars-19.1.0-166 intel-mkl-common-2020.0-166 intel-mkl-core-rt-2020.0-166
  intel-mkl-doc-2020 intel-openmp-19.1.0-166 intel-tbb-libs-2020.0-166
```
```
The following NEW packages will be installed:
  arrayfire-cmake arrayfire-cpu3-dev arrayfire-headers
```


* norm()
https://arrayfire.org/docs/group__lapack__ops__func__norm.htm#gada407977a0136ba855b8bef162dc9fcf

```math
norm(x_1^2 + \cdots + x_n^2) = (x_1^2 + \cdots + x_n^2)^{1/2}
```

$`p = 2`$, $`q = 0.5`$

```
af::array x = af::randu(100, 10);
af::array x_norm(10);

for (size_t i = 0; i < x_norm.dims(0); i++) {
    x_norm(i, 2) = af::norm(x.row(i));
}
```

```
gfor (seq i, x_norm.dims(0)) {
    x_norm(i, 2) = af::norm(x.row(i));
}
```

https://github.com/arrayfire/arrayfire/issues/2843


* reference_trajectory_critic

points = (num_trajectories * num_states, 2)
segments = (num_segments, 2)

(num_trajectories * num_states, num_segments, 2)
or
(num_trajectories * num_states * num_segments, 2)
or
(2, num_trajectories * num_states, num_segments)


distance(point, segment)


# 如何使用git上传文件到github
> 由于自己已经成功把前面的都配置好了所以这个笔记是为了第二次和第二次之后的所有的上传而写的

```commandline
git add .
git commit -m "这里写要提交的话"
git push origin master
```

如果被拒绝，可能是因为不同步，也就是github上出现了新的更新，这时可以：
```commandline
git pull
```





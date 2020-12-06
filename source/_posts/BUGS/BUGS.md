#### VSCODE:支持c++11

Go to `Settings` > `User Settings` In here, search for `Run Code Configuration`:

Under this menu, find: `"code-runner.executorMap"`

Edit this Setting by adding it to User Setting as below for C++11 support:

```shell
"code-runner.executorMap":{
    "cpp": "cd $dir && g++ -std=c++11 $fileName -o $fileNameWithoutExt && $dir$fileNameWithoutExt",
},
```



https://stackoverflow.com/questions/53995830/compiling-c11-in-visual-studio-code
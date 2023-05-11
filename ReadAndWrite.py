import os
class RAW:

    def __init__(self):
        pass

    def Read(self, FilePath):
        with open(FilePath, 'r', encoding='utf-8') as f:
            Text = f.read()
        f.close()
        return Text

    def Write(self, FilePath, Text):
        with open(FilePath, 'w+', encoding='utf-8') as f:
            f.write(Text)
        f.close()

    def AddWrite(self, FilePath, Text):
        with open(FilePath, "a+", encoding="utf-8") as f:
            f.write(Text)
        f.close()

    def IfFolderExists(self, FolderPath):
        if os.path.exists(FolderPath):
            return True
        else:
            return False

    def IfFileExists(self, FilePath):
        if os.path.exists(FilePath):
            return True
        else:
            return False

    def CreateFolder(self, FolderPath):
        if not os.path.exists(FolderPath):
            os.makedirs(FolderPath)


    def CreateFile(self, FilePath):
        if not os.path.exists(FilePath):
            f = open(FilePath, 'w+', encoding='utf-8')
            f.close()
            # 文件会自动关闭
    # 将列表写入文件中
    def WriteListIntoFile(self, FilePath, List):
        # self.CreateFolder(FilePath)
        self.CreateFile(FilePath)
        with open(FilePath, "w+", encoding="utf-8") as f:
            for i in range(0, len(List)):
                f.write(str(List[i])+"\n")

    # 检查文件中是否已经存在某一个行和已知字符相等
    def IfSameLikeOneLine(self, FilePath, Str):
        with open(FilePath, "r", encoding="utf-8") as f:
            LineList = f.readlines()
        if Str+"\n" in LineList:
            return True
        else:
            return False

# raw = RAW()
# raw.WriteListIntoFile("a.txt", ["haode", "干啥"])




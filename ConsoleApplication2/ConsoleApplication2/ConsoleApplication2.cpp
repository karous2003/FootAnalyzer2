#include <iostream>
#include <vector>
#include <stack>
#include <sstream>

using namespace std;

struct coordinate {      //定義一個結構coordinate
    int x;   //橫座標
    int y;   //縱座標
    coordinate(int x, int y) : x(x), y(y) {}    //初始化
};


int main() {
    int m, n;   
    cout << "請輸入迷宮的高(列數):";     //輸入高度
    cin >> m;
    cin.ignore();   //消耗掉換行符\n
    cout << "請輸入迷宮的寬(行數):";     //輸入寬度
    cin >> n;
    cin.ignore();    //消耗掉換行符\n

    vector<vector<int>> matrix(m, vector<int>(n, 0));   //創建一個m列n行的二維向量,初始值都為0

    cout << "請輸入迷宮的元素值（0 表示可走，1 表示不可走）：" << endl;
    for (int i = 0; i < m; i++) {    //遍歷矩陣每一列
        string line;     //定義一個變量line，儲存每一行數據
        getline(cin, line);      //cin讀取整行數據，將其儲存在line中
        istringstream str(line);        //串建一個字串流str，用於分割字串
        for (int j = 0; j < n; j++) {    //遍歷矩陣每一行
            str >> matrix[i][j];        //將從str讀取的元素值存入矩陣中
        }
    }

    stack<coordinate> path; //創建一個堆疊名為path，儲存走過的路徑
    coordinate start = { 0, 0 };    //定義起點座標
    path.push(start);   //將起點座標壓入path中

    // 檢查最後一個座標是否為終點坐標
    coordinate final = path.top();
    if (final.x == m - 1 && final .y == n - 1) {
        cout << "可以走到終點!" << endl;
    }
    else {
        cout << "無法走到終點!" << endl;
    }

    while (!path.empty()) {     //當path不為空時
        coordinate current = path.top();    //取得當前座標
        int x = current.x;      //獲取橫坐標
        int y = current.y;      //獲取縱坐標

        if (x == 0 && y == 0)   //若為原點，值設為2
            matrix[x][y] = 2;

        if (x == m - 1 && y == n - 1)    //若抵達終點，終止迴圈
            break;

        if (x - 1 >= 0 && matrix[x - 1][y] == 0) {      //若往左走不超過邊界，且為未走訪過的路徑
            path.push(coordinate(x - 1, y));    //路徑放入path
            matrix[x - 1][y] = 2;       //該位置設為2
        }
        else if (y - 1 >= 0 && matrix[x][y - 1] == 0) {      //若往上走不超過邊界，且為未走訪過的路徑
            path.push(coordinate(x, y - 1));     //路徑放入path
            matrix[x][y - 1] = 2;         //該位置設為2
        }
        else if (x + 1 < m && matrix[x + 1][y] == 0) {      //若往右走不超過邊界，且為未走訪過的路徑
            path.push(coordinate(x + 1, y));    //路徑放入path
            matrix[x + 1][y] = 2;         //該位置設為2
        }
        else if (y + 1 < n && matrix[x][y + 1] == 0) {      //若往下走不超過邊界，且為未走訪過的路徑
            path.push(coordinate(x, y + 1));    //路徑放入path
            matrix[x][y + 1] = 2;         //該位置設為2
        }
        else {
            matrix[x][y] = 3;   //值設為3
            path.pop();     //刪除前一筆路徑
        }
    }

    cout << "老鼠走過的迷宮：" << endl;
    for (int i = 0; i < m + 2; i++) {         //遍歷每一列
        for (int j = 0; j < n + 2; j++) {     //遍歷每一行
            if (i == 0 || i == m + 1 || j == 0 || j == n + 1)       //若在迷宮外圍
                cout << "1 ";   //生成一圈外牆
            else {
                int value = matrix[i - 1][j - 1];       //獲取迷宮內的元素值
                cout << value << " ";          //輸出元素值
            }
        }
        for (int i = 0; i < m; i++) {       //遍歷每一列
            for (int j = 0; j < n; j++) {     //遍歷每一行
                if (matrix[i][j] == 0) {    //若為未走過的地方
                    matrix[i][j] = 3;       //標記為3
                }
            }
        }
        cout << endl; //換行
    }

    return 0;
}

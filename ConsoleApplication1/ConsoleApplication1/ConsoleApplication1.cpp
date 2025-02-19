#include <iostream>
#include <vector>
#include <stack>

using namespace std;

struct Position {
    int x;
    int y;
};

int main() {
    int m = 5, n = 5; // 預設矩陣大小為5*5

    // 創建一個二維陣列並初始化
    vector<vector<int>> maze(m, vector<int>(n));

    // 使用者輸入迷宮內容
    cout << "請輸入迷宮的元素值：" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << "輸入第 " << i + 1 << " 行第 " << j + 1 << " 列的元素: ";
            cin >> maze[i][j];
        }
    }

    // 創建堆疊，並將起點壓入堆疊中
    stack<Position> pathStack;
    Position start = { 0, 0 }; // 起點為(0, 0)
    pathStack.push(start);

    // 定義四個方向的移動
    int dx[] = { -1, 0, 1, 0 };
    int dy[] = { 0, -1, 0, 1 };

    // 進入迴圈，直到老鼠抵達終點(m-2,n-2)
    while (true) {
        // 提取堆疊頂部的位置作為目前位置
        Position current = pathStack.top();
        int x = current.x;
        int y = current.y;

        // 檢查是否到達終點，如果是，退出迴圈
        if (x == m - 1 && y == n - 1)
            break;

        // 移動四方向
        bool moved = false;
        for (int i = 0; i < 4; i++) {
            int new_x = x + dx[i];
            int new_y = y + dy[i];
            if (new_x >= 0 && new_x < m && new_y >= 0 && new_y < n && maze[new_x][new_y] == 0) {
                // 如果可移動，則將新位置壓入堆疊中
                Position new_position = { new_x, new_y };
                pathStack.push(new_position);
                maze[new_x][new_y] = 2; // 標示為走過的路徑
                moved = true;
                break;
            }
        }

        // 如果四方向都無法移動，則將該位置標記為 3(死路)，並從堆疊中彈出上一步的位置
        if (!moved) {
            maze[x][y] = 3; // 標示為死路
            pathStack.pop();
        }
    }

    // 輸出迷宮及老鼠走過的路徑
    cout << "老鼠走過的路徑：" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (maze[i][j] == 0)
                cout << "  "; // 可走的路徑
            else if (maze[i][j] == 1)
                cout << "1 "; // 牆壁
            else if (maze[i][j] == 2)
                cout << "2 "; // 老鼠走過的路徑
            else if (maze[i][j] == 3)
                cout << "3 "; // 死路
        }
        cout << endl;
    }

    return 0;
}

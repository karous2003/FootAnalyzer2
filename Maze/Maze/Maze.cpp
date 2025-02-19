#include <iostream>
#include <vector>
#include <stack>
#include <sstream>

using namespace std;

struct coordinate {
    int x;
    int y;
    coordinate(int x, int y) : x(x), y(y) {}
};

int main() {
    int m, n;
    cout << "請輸入迷宮的高(列數):";
    cin >> m;
    cin.ignore();
    cout << "請輸入迷宮的寬(行數):";
    cin >> n;
    cin.ignore();

    vector<vector<int>> matrix(m, vector<int>(n, 0));

    cout << "請輸入迷宮的元素值（0 表示可走，1 表示不可走）：" << endl;
    for (int i = 0; i < m; i++) {
        string line;
        getline(cin, line);
        istringstream str(line);
        for (int j = 0; j < n; j++) {
            str >> matrix[i][j];
        }
    }

    stack<coordinate> path;
    coordinate start = { 0, 0 };
    path.push(start);

    while (!path.empty()) {
        coordinate current = path.top();
        int x = current.x;
        int y = current.y;

        if (x == 0 && y == 0)
            matrix[x][y] = 2;

        if (x == m - 1 && y == n - 1)
            break;

        if (x - 1 >= 0 && matrix[x - 1][y] == 0) {
            path.push(coordinate(x - 1, y));
            matrix[x - 1][y] = 2;
        }
        else if (y - 1 >= 0 && matrix[x][y - 1] == 0) {
            path.push(coordinate(x, y - 1));
            matrix[x][y - 1] = 2;
        }
        else if (x + 1 < m && matrix[x + 1][y] == 0) {
            path.push(coordinate(x + 1, y));
            matrix[x + 1][y] = 2;
        }
        else if (y + 1 < n && matrix[x][y + 1] == 0) {
            path.push(coordinate(x, y + 1));
            matrix[x][y + 1] = 2;
        }
        else {
            matrix[x][y] = 3;
            path.pop();
        }
    }

    cout << "老鼠走過的迷宮：" << endl;
    for (int i = 0; i < m + 2; i++) {
        for (int j = 0; j < n + 2; j++) {
            if (i == 0 || i == m + 1 || j == 0 || j == n + 1)
                cout << "1 ";
            else {
                int value = matrix[i - 1][j - 1];
                cout << value << " ";
            }
        }
        cout << endl;
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] == 0) {
                matrix[i][j] = 3;
            }
        }
    }
    return 0;
}

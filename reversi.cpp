#include <utility>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

#define BOARD_SIZE 8
#define EMPTY 0
#define ME 1
#define OTHER 2
#define MAX_DEPTH 6
#define INF 2100000000

#define START "START"
#define PLACE "PLACE"
#define DONE  "DONE"
#define TURN  "TURN"
#define BEGIN "BEGIN"
#define END   "END"
#define PASS  "PASS"

struct Position {
    int x;
    int y;
};

ostream &operator<<(ostream &out, const Position &pos) {
    return out << (pos.x) << " " << (pos.y);
}

struct _movement{
    int x, y, steps;
    pair<int, int> dir;
};

class AI {
private:
    /*
     * You can define your own struct and variable here
     * 你可以在这里定义你自己的结构体和变量
     */
    size_t boardSize;
    vector<vector<int> > &board;
    const vector<pair<int, int> > directions = {
            {1,  -1},
            {1,  0},
            {1,  1},
            {-1, -1},
            {-1, 0},
            {-1, 1},
            {0,  -1},
            {0,  1}
    };
    Position preferedPos;
public:
    explicit AI(vector<vector<int> > &board);

    void init();

    void Display();
    
    int Evaluate();

    unsigned long long hash();

    int Search(int depth, int alpha, int beta, int CurrentPlayer);

    Position begin();

    Position turn(const Position &other);

    bool canrev(int i, int j, int who);

    int reverseDirection(int i, int j, int who, pair<int, int> dir, bool reverse = false);

    void undoreverseDirection(int i, int j, int who, pair<int, int> dir, int steps);

    bool putChess(int i, int j, int who);
};

AI::AI(vector<vector<int> > &board) : board(board) {
    this->boardSize = board.size();
}

class REPL {
private:
    vector<vector<int> > board;
    AI ai;

    const char symbol[3] = {' ', 'X', 'O'};
public:
    REPL();

    ~REPL() = default;

    void printBoard();

    void loop();
};

REPL::REPL() : ai(board) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        board.emplace_back(move(vector<int>(BOARD_SIZE, 0)));
    }
}

void REPL::printBoard() {
    for (int i = 0; i < BOARD_SIZE; i++) {
        fprintf(stderr, "   %c", 'a' + i);
    }
    for (int i = 0; i < BOARD_SIZE; i++) {
        fprintf(stderr, "\n ");
        for (int j = 0; j < BOARD_SIZE; j++) {
            fprintf(stderr, "+---");
        }
        fprintf(stderr, "+\n%c", '1' + i);
        for (int j = 0; j < BOARD_SIZE; j++) {
            fprintf(stderr, "| %c ", symbol[board[i][j]]);
        }
        fprintf(stderr, "|");
    }
    fprintf(stderr, "\n ");
    for (int i = 0; i < BOARD_SIZE; i++) {
        fprintf(stderr, "+---");
    }
    fprintf(stderr, "+\n");
}

void REPL::loop() {
    while (true) {
        string buffer;
        getline(cin, buffer);
        if (buffer.empty()) continue;
        istringstream ss(buffer);

        string command;
        ss >> command;
        if (command == START) {
            ai.init();
        } else if (command == PLACE) {
            int x, y, z;
            char temp;
            ss >> temp >> y >> z;
            x = temp - 'a';
            --y;
            board[y][x] = z;
        } else if (command == DONE) {
            // cout << "OK" << endl;
        } else if (command == BEGIN) {
            Position pos = ai.begin();
            ai.putChess(pos.x, pos.y, ME);
            // cout << pos << endl;
            // ai.Display();
            return;
        } else if (command == TURN) {
            int x, y;
            char temp;
            ss >> temp >> y;
            x = temp - 'a';
            --y;
            if (ai.putChess(y, x, OTHER)) {
                Position pos = ai.turn({x, y});
                if (pos.x >= 0) {
                    ai.putChess(pos.x, pos.y, ME);
                    cout << pos << endl;
                } else {
                    cout << PASS << endl;
                }
            } else {
                cout << "input error" << endl;
            }
        } else if (command == PASS) {
            Position pos = ai.turn({0, 0});
            if (pos.x >= 0) {
                ai.putChess(pos.x, pos.y, ME);
                cout << pos << endl;
            } else {
                cout << PASS << endl;
            }
        } else if (command == END) {
            break;
        }
        //printBoard();
    }
}

void AI::Display(){
    char meaning[]={' ', 'x', 'o'}; 
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            printf("%c", meaning[board[i][j]]);
        }
        printf("\n");
    }
}

/*
 * YOUR CODE BEGIN
 * 你的代码开始
 */

bool AI::putChess(int i, int j, int who) {
    if (i < 0 || i >= boardSize || j < 0 || j >= boardSize) return false;
    if (board[i][j] != EMPTY) return false;
    bool flag = false;
    for (auto &dir: directions) {
        int num = reverseDirection(i, j, who, dir);
        if (num) {
            flag = true;
            reverseDirection(i, j, who, dir, true);
        }
    }
    if (flag) {
        board[i][j] = who;
    }
    return flag;
}

int AI::reverseDirection(int i, int j, int who, pair<int, int> dir, bool reverse) {
    int x = i + dir.first;
    int y = j + dir.second;
    int num = 0;
    bool success = false;
    int other = 3 - who;
    while (x >= 0 && x < boardSize && y >= 0 && y < boardSize) {
        if (board[x][y] == other) {
            num++;
            if (reverse) board[x][y] = who;
        } else if (board[x][y] == who && num > 0) {
            success = true;
            break;
        } else {
            break;
        }
        x += dir.first;
        y += dir.second;
    }
    if (!success) return 0;
    return num;
}

void AI::undoreverseDirection(int i, int j, int who, pair<int, int> dir, int steps){
    // printf("%d %d %d %d\n",i,j,who,steps);
    int x = i + dir.first;
    int y = j + dir.second;
    board[i][j]=0;
    for(int i=0; i<steps; i++){
        board[x][y] = 3 - board[x][y];
        x += dir.first;
        y += dir.second;
    }
}

bool AI::canrev(int i, int j, int who) {
    if (i < 0 || i >= boardSize || j < 0 || j >= boardSize) return false;
    if (board[i][j] != EMPTY) return false;
    for (auto &dir: directions) {
        int num = reverseDirection(i, j, who, dir);
        if (num) return true;
    }
    return false;
}

/*
 * You should init your AI here
 * 在这里初始化你的AI
 */
void AI::init() {
    this->boardSize = board.size();
}

int AI::Evaluate(){
    int emptyCount = 0;
    int score = 0, attr[]={0,1,-1};
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if(board[i][j]==0)emptyCount++;
            score += attr[board[i][j]];
        }
    }
    if(emptyCount == 0) return score;
    int LMPos=0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (EMPTY == board[i][j] && canrev(i, j, 1)) {
                LMPos++;
            }
        }
    }
    score += LMPos * 100;
    // for (int i = 0; i < BOARD_SIZE; i++) {
    score += 1000 * attr[board[0][0]];
    score += 1000 * attr[board[BOARD_SIZE-1][0]];
    score += 1000 * attr[board[0][BOARD_SIZE-1]];
    score += 1000 * attr[board[BOARD_SIZE-1][BOARD_SIZE-1]];
    // }

    return score;
}


// cite: http://www.xqbase.com/computer/search_alphabeta.htm
int AI::Search(int depth, int alpha, int beta, int CurrentPlayer = 1){
    if (depth == 0){
        int eval = Evaluate();
        // printf("Eval: %d\n", eval);
        // Display();
        return eval;
    }

    // Generate Moves
    int LegalMoves[64][2], LMPos=0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (EMPTY == board[i][j] && canrev(i, j, CurrentPlayer)) {
                LegalMoves[LMPos][0] = i;
                LegalMoves[LMPos][1] = j;
                // printf("Legal move: %d %d\n",i,j);
                LMPos++;
            }
        }
    }
    if(!LMPos){
        return -Search(depth - 1, -beta, -alpha, 3 - CurrentPlayer);
    }
    while (LMPos--){
        int x = LegalMoves[LMPos][0];
        int y = LegalMoves[LMPos][1];

        _movement moves[64];
        int movesPos = 0;

        // putChess
        int i=x, j=y, who=CurrentPlayer;
        for (auto &dir: directions) {
            int num = reverseDirection(i, j, who, dir);
            if (num) {
                moves[movesPos].steps = reverseDirection(i, j, who, dir, true);
                // record moves
                moves[movesPos].x = i;
                moves[movesPos].y = j;
                moves[movesPos].dir=dir;
                movesPos++;
            }
        }
        board[x][y] = who;

        int val = -Search(depth - 1, -beta, -alpha, 3 - CurrentPlayer);
        if(depth == MAX_DEPTH){
            printf("%d %d %d\n", x, y, val);
            Display();
        }
        
        // unputChess
        while(movesPos--){
            undoreverseDirection(
                moves[movesPos].x,
                moves[movesPos].y,
                who,
                moves[movesPos].dir,
                moves[movesPos].steps
            );
        }

        if (val >= beta){
            return beta;
        }
        if (val > alpha){
            if(depth == MAX_DEPTH){
                preferedPos = {x, y};
            }
            alpha = val;
        }
    }
    return alpha;
}


/*
 * Game Start, you will put the first chess.
 * Warning: This method will only be called when after the initialize of the map,
 * it is your turn to put the chess, or this method will not be called.
 * You should return a valid Position variable.
 * 棋局开始，首先由你来落子
 * 请注意：只有在当棋局初始化后，轮到你落子时才会触发这个函数。如果在棋局初始化完毕后，轮到对手落子，则这个函数不会被触发。详见项目要求。
 * 你需要返回一个结构体Position，在x属性和y属性填上你想要落子的位置。
 */



Position AI::begin() {
    return turn({0, 0});
}

unsigned long long AI::hash() {
    unsigned long long hash = 0;
    for (int i = 1; i < BOARD_SIZE - 1; i++) {
        for (int j = 1; j < BOARD_SIZE - 1; j++) {
            hash *= 3;
            hash += board[i][j];
        }
    }
    return hash;
}

Position AI::turn(const Position &other) {
    /*
     * TODO: Write your own ai here!
     * Here is a simple AI which just put chess at empty position!
     * 代做：在这里写下你的AI。
     * 这里有一个示例AI，它只会寻找第一个可下的位置进行落子。
     */

    // cout<<hash()<<endl;
    // unsigned long long hash = hash();
    // switch(hash()){
    //     case 15279122725965:
    //         return {2, 2};
    //     case 90689875209:
    //         return {2, 2};
    //     case 17472198879:
    //         return {5, 5};
    //     case 17475374403:
    //         return {5, 5};
    // }
    int i, j;
    preferedPos = {-1, -1};
    // Display();
    Search(MAX_DEPTH, -INF, INF, ME);
    // Display();
    return preferedPos;
}

/*
 * YOUR CODE END
 * 你的代码结束
 */

int main(int argc, char *argv[]) {
    REPL repl;
    repl.loop();
    return 0;
}

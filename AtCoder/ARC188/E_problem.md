E - Gravity Sort [Editorial](https://atcoder.jp/contests/arc188/tasks/arc188_e/editorial) ![](https://img.atcoder.jp/assets/top/img/flag-lang/ja.png) / ![](https://img.atcoder.jp/assets/top/img/flag-lang/en.png) $(function() { var ts = $('#task-statement span.lang'); if (ts.children('span').size() <= 1) { $('#task-lang-btn').hide(); ts.children('span').show(); return; } var REMEMBER\_LB = 5; var LS\_KEY = 'task\_lang'; var taskLang = getLS(LS\_KEY) || ''; var changeTimes = 0; if (taskLang == 'ja' || taskLang == 'en') { changeTimes = REMEMBER\_LB; } else { var changeTimes = parseInt(taskLang, 10); if (isNaN(changeTimes)) { changeTimes = 0; delLS(LS\_KEY); } changeTimes++; taskLang = LANG; } ts.children('span.lang-' + taskLang).show(); $('#task-lang-btn span').click(function() { var l = $(this).data('lang'); ts.children('span').hide(); ts.children('span.lang-' + l).show(); if (changeTimes < REMEMBER\_LB) setLS(LS\_KEY, changeTimes); else setLS(LS\_KEY, l); }); });

* * *

Time Limit: 2 sec / Memory Limit: 1024 MB

配点 : 100010001000 点

### 問題文

111 から 2N2N2N までの番号がついた 2N2N2N 個のマスが、マス 111 を上として上下一列に並んでいます。各マスには 111 つずつボールが入っており、時刻 t\=0t=0t\=0 において、マス iii にあるボールの重さは、 i\=1,2,…,Ni=1,2, \\ldots ,Ni\=1,2,…,N では mim\_imi​、 i\=N+1,N+2,…,2Ni=N+1,N+2, \\ldots ,2Ni\=N+1,N+2,…,2N では 000 です。ただし、(m1,m2,…,mN)(m\_1, m\_2, \\ldots , m\_N)(m1​,m2​,…,mN​) は 111 から NNN までの整数を並び替えた数列です。

以下では、重さ iii のボールを「ボール iii」、各ボールの入っているマスの番号を「ボールの位置」と呼ぶことにします。

時刻 t\=0t=0t\=0 以降、時刻が 111 進むごとに、重いボールが下に沈み、代わりに軽いボールが浮かび上がっていきます。

厳密には、以下の手順によって、時刻 t\=t0t=t\_0t\=t0​ における各ボールの位置から時刻 t\=t0+1t=t\_0+1t\=t0​+1 における各ボールの位置を定めます。

> -   まず、i\=N,N−1,…,2,1i=N,N-1,\\ldots ,2,1i\=N,N−1,…,2,1 の順に、以下の操作を行う。
>     -   ボール iii の t\=t0+1t=t\_0+1t\=t0​+1 における位置がすでに定められている場合
>         -   **何もしない。**
>     -   ボール iii の t\=t0+1t=t\_0+1t\=t0​+1 における位置がまだ定められていない場合
>         -   t\=t0t=t\_0t\=t0​ にボール iii の 111 つ下のマスが存在し、このマスに入っているボール（ボール jjj とする）がボール iii より軽いとき、**ボール iii と ボール jjj の t\=t0+1t=t\_0+1t\=t0​+1 における位置を、t\=t0t=t\_0t\=t0​ における両者の位置を交換したものとして定める。**
>         -   上記にあてはまらないとき、**ボール iii の t\=t0+1t=t\_0+1t\=t0​+1 における位置を t\=t0t=t\_0t\=t0​ における位置と同じに定める。**
> -   続いて、ボール 000 のうちこの時点で t\=t0+1t=t\_0+1t\=t0​+1 における位置が定められていないものについて、**それぞれ t\=t0+1t=t\_0+1t\=t0​+1 における位置を t\=t0t=t\_0t\=t0​ における位置と同じに定める。**

このとき、ある時刻にボールが上から軽い順に並び、それ以降全てボールの位置が変化しなくなることが示せます。この状態に達する時刻を求めてください。

### 制約

-   1≤N≤2×1051\\leq N\\leq 2\\times 10^51≤N≤2×105
-   1≤mi≤N1\\leq m\_i \\leq N1≤mi​≤N
-   i≠ji\\neq ji\=j のとき mi≠mjm\_i\\neq m\_jmi​\=mj​
-   入力される値はすべて整数である

* * *

### 入力

入力は以下の形式で標準入力から与えられる。

NNN
m1m\_1m1​ m2m\_2m2​ …\\ldots… mNm\_NmN​

### 出力

答えを整数で出力せよ。

* * *

### 入力例 1Copy

Copy

3
2 3 1

### 出力例 1Copy

Copy

6

時刻 t\=0t=0t\=0 から t\=1t=1t\=1 にかけての動きは次のように定まります。（必要に応じて、下の図も参考にしてください。）

> 1.  ボール 333 について、t\=1t=1t\=1 における位置はまだ定められていない。111 つ下のマスにはボール 111 があり、ボール 333 の方が重いため、t\=1t=1t\=1 には両者の位置を入れ替える。すなわち、ボール 333 の位置をマス 333、ボール 111 の位置をマス 222 に定める。
> 2.  ボール 222 について、t\=1t=1t\=1 における位置はまだ定められていない。111 つ下のマスにはボール 333 があり、これはボール 222 より重いため、ボール 222 の t\=1t=1t\=1 における位置を t\=0t=0t\=0 と同じに定める。
> 3.  ボール 111 について、t\=1t=1t\=1 における位置は先のステップで既に定められている。
> 4.  ボール 000 について、全て t\=1t=1t\=1 における位置はまだ定められていない。これらは全て t\=1t=1t\=1 での位置を t\=0t=0t\=0 と同じに定める。

続いて、時刻 t\=1t=1t\=1 から t\=2t=2t\=2 にかけての動きは次のように定まります。

> 1.  ボール 333 について、t\=2t=2t\=2 における位置はまだ定められていない。111 つ下のマスにはボール 000 があり、ボール 333 の方が重いため、t\=2t=2t\=2 には両者の位置を入れ替える。すなわち、ボール 333 の位置をマス 444、（ボール 333 の 111 つ下にあった）ボール 000 の位置をマス 333 に定める。
> 2.  ボール 222 について、t\=2t=2t\=2 における位置はまだ定められていない。111 つ下のマスにはボール 111 があり、ボール 222 の方が重いため、t\=2t=2t\=2 には両者の位置を入れ替える。すなわち、ボール 222 の位置をマス 222、ボール 111 の位置をマス 111 に定める。
> 3.  ボール 111 について、t\=2t=2t\=2 における位置は先のステップで既に定められている。
> 4.  ボール 000 について、t\=1t=1t\=1 にマス 444 にあったものの t\=2t=2t\=2 における位置は先のステップで既に定められている。それ以外について、t\=2t=2t\=2 での位置を t\=1t=1t\=1 と同じに定める。

この後も同様にボールの位置を定めていくと、時刻 t\=6t=6t\=6 に上から順にボール 0,0,0,1,2,30,0,0,1,2,30,0,0,1,2,3 が並び、以降ボールの位置が変化しないことが分かります。

![](https://img.atcoder.jp/arc188/4e545d6825157293f80acafb7314f5d1.png)

* * *

### 入力例 2Copy

Copy

5
4 1 2 3 5

### 出力例 2Copy

Copy

9

* * *

### 入力例 3Copy

Copy

1
1

### 出力例 3Copy

Copy

1
Score : 100010001000 points

### Problem Statement

There are 2N2N2N cells numbered from 111 to 2N2N2N arranged vertically in a column with cell 111 at the top. Each cell contains one ball. The weight of the ball in cell iii at time t\=0t=0t\=0 is mim\_imi​ for i\=1,2,…,Ni=1,2,\\ldots,Ni\=1,2,…,N, and 000 for i\=N+1,N+2,…,2Ni=N+1,N+2,\\ldots,2Ni\=N+1,N+2,…,2N. Here, (m1,m2,…,mN)(m\_1, m\_2, \\ldots, m\_N)(m1​,m2​,…,mN​) is a permutation of the integers from 111 to NNN.

In the following, we will refer to the ball of weight iii as ball iii, and the cell number where each ball is located as the position of the ball.

From time t\=0t=0t\=0 onwards, every time the time advances by 111, the heavier balls sink downward, and the lighter balls rise upward.

Formally, the positions of each ball at time t\=t0+1t=t\_0+1t\=t0​+1 are determined from their positions at time t\=t0t=t\_0t\=t0​ by the following procedure.

> -   First, for i\=N,N−1,…,2,1i=N,N-1,\\ldots,2,1i\=N,N−1,…,2,1 in this order, perform the following operation.
>     -   If the position of ball iii at t\=t0+1t=t\_0+1t\=t0​+1 has already been determined:
>         -   **Do nothing.**
>     -   If the position of ball iii at t\=t0+1t=t\_0+1t\=t0​+1 has not been determined:
>         -   If there exists a cell immediately below ball iii at t\=t0t=t\_0t\=t0​, and the ball in that cell (call it ball jjj) is lighter than ball iii, **set the positions of balls iii and jjj at t\=t0+1t=t\_0+1t\=t0​+1 to be swapped from their positions at t\=t0t=t\_0t\=t0​**.
>         -   Otherwise, **set the position of ball iii at t\=t0+1t=t\_0+1t\=t0​+1 to be the same as at t\=t0t=t\_0t\=t0​.**
> -   Next, for all balls of weight 000 whose positions at t\=t0+1t=t\_0+1t\=t0​+1 have not been determined at this point, **set their positions at t\=t0+1t=t\_0+1t\=t0​+1 to be the same as at t\=t0t=t\_0t\=t0​.**

It can be shown that at some time, the balls will be arranged from top to bottom in ascending order of weight, and their positions will no longer change. Find the time when this state is reached.

### Constraints

-   1≤N≤2×1051 \\leq N \\leq 2 \\times 10^51≤N≤2×105
-   1≤mi≤N1 \\leq m\_i \\leq N1≤mi​≤N
-   mi≠mjm\_i \\neq m\_jmi​\=mj​ for i≠ji \\neq ji\=j.
-   All input values are integers.

* * *

### Input

The input is given from Standard Input in the following format:

NNN
m1m\_1m1​ m2m\_2m2​ …\\ldots… mNm\_NmN​

### Output

Print the answer as an integer.

* * *

### Sample Input 1Copy

Copy

3
2 3 1

### Sample Output 1Copy

Copy

6

The movements from time t\=0t=0t\=0 to t\=1t=1t\=1 are determined as follows. (Refer to the diagram below if necessary.)

> 1.  For ball 333, its position at t\=1t=1t\=1 has not yet been determined. The cell immediately below it contains ball 111, and ball 333 is heavier, so swap their positions for t\=1t=1t\=1. That is, set the position of ball 333 to cell 333, and ball 111 to cell 222.
> 2.  For ball 222, its position at t\=1t=1t\=1 has not yet been determined. The cell immediately below it contains ball 333, which is heavier than ball 222, so set the position of ball 222 at t\=1t=1t\=1 to be the same as at t\=0t=0t\=0.
> 3.  For ball 111, its position at t\=1t=1t\=1 has already been determined in the earlier step.
> 4.  For balls of weight 000, none of their positions at t\=1t=1t\=1 have been determined. Set their positions at t\=1t=1t\=1 to be the same as at t\=0t=0t\=0.

Next, the movements from time t\=1t=1t\=1 to t\=2t=2t\=2 are determined as follows.

> 1.  For ball 333, its position at t\=2t=2t\=2 has not yet been determined. The cell immediately below it contains ball 000, and ball 333 is heavier, so swap their positions for t\=2t=2t\=2. That is, set the position of ball 333 to cell 444, and ball 000 (the one that was below ball 333) to cell 333.
> 2.  For ball 222, its position at t\=2t=2t\=2 has not yet been determined. The cell immediately below it contains ball 111, and ball 222 is heavier, so swap their positions for t\=2t=2t\=2. That is, set the position of ball 222 to cell 222, and ball 111 to cell 111.
> 3.  For ball 111, its position at t\=2t=2t\=2 has already been determined in the earlier step.
> 4.  For balls of weight 000, the one that was at cell 444 at t\=1t=1t\=1 has already had its position at t\=2t=2t\=2 determined in the earlier step. For the others, set their positions at t\=2t=2t\=2 to be the same as at t\=1t=1t\=1.

Continuing to determine the positions of the balls in this way, at time t\=6t=6t\=6, the balls will be arranged from top to bottom as balls 0,0,0,1,2,30,0,0,1,2,30,0,0,1,2,3, and their positions will no longer change.

![](https://img.atcoder.jp/arc188/4e545d6825157293f80acafb7314f5d1.png)

* * *

### Sample Input 2Copy

Copy

5
4 1 2 3 5

### Sample Output 2Copy

Copy

9

* * *

### Sample Input 3Copy

Copy

1
1

### Sample Output 3Copy

Copy

1

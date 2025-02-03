#include <bits/stdc++.h>
using namespace std;
 
typedef long long ll;
 
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
 
    // Read the number of weighted balls (top N cells have balls with weights 1..N)
    int n;
    cin >> n;
 
    // Array 'pos' will hold the starting (0-indexed) position of each weighted ball.
    // For a ball with weight x, we store its index (i) as pos[x-1].
    vector<int> pos(n);
    for (int i = 0; i < n; i++){
        int x;
        cin >> x;
        pos[x - 1] = i;
    }
 
    // Create a set of available offset tokens.
    // We insert all integers from -n to 2*n. These tokens represent possible extra delays.
    // The analysis of the process shows that the useful tokens lie in this interval.
    set<int> tokens;
    for (int x = -n; x <= 2 * n; x++){
        tokens.insert(x);
    }
 
    int ans = 0; // This will store the final answer (i.e. the number of rounds needed until stabilization)
 
    // Process the balls in descending order of weight.
    // We loop from i = n-1 down to 0.
    // Here, i is 0-indexed for our 'pos' array and corresponds to the ball with weight (i+1).
    for (int i = n - 1; i >= 0; i--){
        // For the current ball, its starting (0-indexed) position is pos[i].
        // We need to select an offset token x (from our set) that satisfies:
        //     x >= - (starting position)
        // The lower_bound function gives the smallest token not less than -pos[i].
        auto it = tokens.lower_bound(-pos[i]);
        if(it == tokens.end()){
            // In practice, this should never happen given our token range
            continue;
        }
        int token = *it;
 
        // If token is nonnegative, it contributes directly to the stabilization time.
        // The candidate stabilization time for this ball is:
        //      candidate = token + (i + n)
        // where (i + n) accounts for the ball's drop distance and its ordering.
        if (token >= 0) {
            ans = max(ans, token + i + n);
        }
 
        // Remove the chosen token as well as its immediate neighbors.
        // This is needed to enforce that delay tokens used by different balls remain "separated"
        // by at least 1, in accordance with the detailed analysis.
        tokens.erase(token - 1);
        tokens.erase(token);
        tokens.erase(token + 1);
    }
 
    cout << ans << "\n";
    return 0;
}

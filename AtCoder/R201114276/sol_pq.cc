#include <iostream>
#include <queue>
using namespace std;

priority_queue<long long> neg_pq;
long long n,k;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n >> k;
    long long price;

    for (auto i = 0; i < n; i++)
    {
        cin >> price;
        neg_pq.push(-price);
    }

    n = 0; // set n to 0 to store the sum

    for (auto i = 0; i < k; i++)
    {
        n -= i;
        n += neg_pq.top();
        neg_pq.pop();
    }
 
    cout << -n << "\n";
 
    return 0;
}

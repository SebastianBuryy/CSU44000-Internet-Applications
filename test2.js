"use strict";

function callmebackin(s, subject) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve("Returned call subject: " + subject)
        }, s * 1000)
    })
}

// trying out Async Await
async function waitaround() {
    let topic1 = await callmebackin(5, "about a dog");
    console.log("Got called back: " + topic1)
    let topic2 = await callmebackin(3, "about another dog");
    console.log("You called me back: " + topic2)
}

waitaround()
console.log("Hoping to get some calls")
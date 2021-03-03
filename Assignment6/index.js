var prevDate = new Date();
var totalInvocations = 0;
var seconds = 0;
var average = 0;

exports.handler = async (event) => {
    var currDate = new Date();    
    var currInvocation = currDate.toJSON();

    var reset = false;
    if (event.queryStringParameters !== null && event.queryStringParameters !== undefined) {
        if (event.queryStringParameters["cmd"] == "RESET") {
            totalInvocations = 0;
            seconds = 0;
            average = 0;
            reset = true;
        }
    }
    
    if (totalInvocations == 0) {
        prevDate = currDate;
    }
    else {
        var diff = currDate.getTime() - prevDate.getTime();
        seconds = Math.floor(diff/1000);
    };
   
    if (totalInvocations == 0) {
        average = 0;
    }
    else if (totalInvocations == 1) {
        average = seconds;
    }
    else {
        average = (seconds + average * (totalInvocations-1))/totalInvocations;
    };
     
    totalInvocations++;
    prevDate = currDate;    

    var res1 = {
        "ThisInvocation" : currInvocation,
        "TimeSinceLast" : seconds,
        "TotalInvocationsOnThisContainer" : totalInvocations,
        "AverageGapBetweenInvocations" : average
    };
    
    var res2 = {
        "ThisInvocation" : currInvocation
        
    };
    
    var res;
    
    if (reset == true) {
        res = res2
    } else {
        res = res1
    }
    
    const response = {
        body: res
    };
    
    return response;
};
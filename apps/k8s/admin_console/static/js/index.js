$(document).ready(function () {

    // Modal box of comment
    $('#identifier').modal()

    // Click the cancel button
    $(".cancel_job").click(function () {
        $("#comment_text").val("");
        $('#myModal').modal("hide")
    })

    // Click the x button
    $(".close").click(function () {
        $("#comment_text").val("");
        $('#myModal').modal("hide")
    })

    // Click the submit button
    // TO do: A job_id is required
    $(".comment_job").click(function () {
        var comment = $("#comment_text").val();
        var data = {
            data: JSON.stringify({"comment": comment}),
        }
        $.ajax({
            url: "http://usa-data.baidu.com:8001/api/v1/namespaces/default/services/http:admin-console-service:8000/proxy/submit_job",
            dataType: "json",
            type: "POST",
            data: data,
            success: function (result) {
                $("#comment_text").val("");
                $('#myModal').modal("hide");
                var a = JSON.stringify(result)
                alert(a);
            },
            error: function (xhr, textStatus, errorThrown) {
                alert("submit failure")
            }
        })
    })

    // Click the expand
    $(".job_expand").click(function () {
        $(this).parent().parent().nextUntil(".flag").css("display", "table-row")
    })

    // Click the collapse
    $(".job_collapse").click(function () {
        $(this).parent().parent().nextUntil(".flag").css("display", "none")
    })
})
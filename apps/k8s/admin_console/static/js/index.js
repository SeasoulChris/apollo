$(document).ready(function () {

    // submit the job-type select
    $('#type-select').change(function () {
        $("#job-type-form").submit();
    });

    // submit the time-field select
    $('#time-select').change(function () {
        $("#time-form").submit();
    });

    // submit the search by vehicle SN
    $("#search-btn").click(function () {
        $("#search-form").submit();
    });

    // Modal box of comment
    $('#identifier').modal()

    // Click the body
    $('body').click(function () {
        $("#comment_text").val("");
    });

    // Click the job-modal
    $(".job-modal").click(function () {
        var job_id = $(this).attr("id");
        var action = $(this).val();
        $(this).append('<div id=' + job_id + ' class="job-id"> </div><div id="job-action" class=' + action + '> </div>');
        $("#myModalLabel").text(action + " Job");
        $(".comment-title").text("Do you want to " + action + " this Job?")
    })

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
    $(".comment_job").click(function () {
        var comment = $("#comment_text").val();
        var job_id = $(".job-id").attr("id");
        var action = $("#job-action").attr("class");
        var job_data = {"comment": comment, "job_id": job_id, "action": action};
        $.ajax({
            url: "http://usa-data.baidu.com:8001/api/v1/namespaces/default/services/http:admin-console-service:8000/proxy/submit_job",
            dataType: "json",
            type: "POST",
            data: job_data,
            success: function (result) {
                $("#comment_text").val("");
                $('#myModal').modal("hide");
                var a = JSON.stringify(result)
                alert(a);
                location.reload();
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
